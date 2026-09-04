#include "pair_transformers_ace_aot.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace LAMMPS_NS;

PairTransformersACEAOT::PairTransformersACEAOT(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairTransformersACEAOT::~PairTransformersACEAOT()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairTransformersACEAOT::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairTransformersACEAOT::settings(int narg, char **arg)
{
  device_spec_ = "auto";
  if (narg == 0) return;
  if (narg == 2 && strcmp(arg[0], "device") == 0) {
    device_spec_ = arg[1];
    return;
  }
  error->all(FLERR,
             "Illegal pair_style transformers_ace/aot command. Use: "
             "pair_style transformers_ace/aot [device auto|cpu|cuda|cuda:N]");
}

std::vector<std::string> PairTransformersACEAOT::split_words(const std::string &line) const
{
  std::stringstream stream(line);
  std::vector<std::string> words;
  std::string word;
  while (stream >> word) words.push_back(word);
  return words;
}

std::map<std::string, std::string>
PairTransformersACEAOT::parse_metadata(const std::string &text) const
{
  std::map<std::string, std::string> fields;
  std::stringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    const auto pos = line.find('=');
    if (pos == std::string::npos) continue;
    fields[line.substr(0, pos)] = line.substr(pos + 1);
  }
  return fields;
}

int PairTransformersACEAOT::local_rank() const
{
  for (const char *name : {"OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
                           "SLURM_LOCALID", "LOCAL_RANK"}) {
    const char *value = std::getenv(name);
    if (value != nullptr) return std::atoi(value);
  }
  return comm->me;
}

torch::Device PairTransformersACEAOT::resolve_device() const
{
  if (device_spec_ == "cpu") return torch::Device(torch::kCPU);
  if (device_spec_.rfind("cuda", 0) == 0) {
    if (!torch::cuda::is_available())
      error->all(FLERR, "pair_style transformers_ace/aot requested CUDA but LibTorch reports none");
    if (device_spec_ == "cuda") {
      const int ndev = static_cast<int>(torch::cuda::device_count());
      return torch::Device(torch::kCUDA, ndev > 0 ? local_rank() % ndev : 0);
    }
    return torch::Device(device_spec_);
  }
  if (device_spec_ != "auto")
    error->all(FLERR, "pair_style transformers_ace/aot device must be auto, cpu, cuda or cuda:N");
  if (torch::cuda::is_available()) {
    const int ndev = static_cast<int>(torch::cuda::device_count());
    return torch::Device(torch::kCUDA, ndev > 0 ? local_rank() % ndev : 0);
  }
  return torch::Device(torch::kCPU);
}

torch::Tensor PairTransformersACEAOT::cell_tensor() const
{
  // Row-vector cell, matching Eq. (4): r_ij = r_j - r_i + S_ij h.
  auto cell = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32));
  auto acc = cell.accessor<float, 2>();
  acc[0][0] = static_cast<float>(domain->xprd);
  acc[1][0] = static_cast<float>(domain->xy);
  acc[1][1] = static_cast<float>(domain->yprd);
  acc[2][0] = static_cast<float>(domain->xz);
  acc[2][1] = static_cast<float>(domain->yz);
  acc[2][2] = static_cast<float>(domain->zprd);
  return cell;
}

void PairTransformersACEAOT::load_package(const std::string &path)
{
  try {
    const int index = device_.is_cuda() ? device_.index() : -1;
    loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        path, "model", /*run_single_threaded=*/false, /*num_runners=*/1, index);
  } catch (const std::exception &err) {
    std::string message = "Could not load TRACE AOTI package: " + path + " (" + err.what() + ")";
    error->all(FLERR, message.c_str());
  }
  model_loaded_ = true;
}

void PairTransformersACEAOT::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR, "pair_coeff must be: * * model.pt2 type1 type2 ...");
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff transformers_ace/aot requires leading '* *'");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 0;

  const std::string model_path(arg[2]);
  device_ = resolve_device();
  load_package(model_path);

  // aot_deploy writes the contract next to the package as <model>.metadata.txt.
  const std::string meta_path = model_path + ".metadata.txt";
  std::ifstream meta_stream(meta_path);
  if (!meta_stream)
    error->all(FLERR, ("Missing AOTI metadata file: " + meta_path).c_str());
  std::stringstream buffer;
  buffer << meta_stream.rdbuf();
  const auto metadata = parse_metadata(buffer.str());

  for (const char *key : {"r_max", "type_symbols", "type_atomic_numbers", "max_atoms", "max_edges"})
    if (!metadata.count(key))
      error->all(FLERR, ("TRACE AOTI metadata is missing field: " + std::string(key)).c_str());

  // The metadata is the contract between aot_deploy and this pair style: units,
  // padding scheme, and force/virial sign conventions all travel through it. A
  // future format revision that silently changed any of those would produce
  // plausible-looking but wrong dynamics, so refuse anything unrecognised.
  const std::string expected_format = "transformers_ace_aoti_v1";
  const auto format = metadata.find("format");
  if (format == metadata.end() || format->second != expected_format)
    error->all(FLERR, ("TRACE AOTI metadata format mismatch: expected " + expected_format +
                       ", got " + (format == metadata.end() ? "<missing>" : format->second) +
                       ". Re-export the model with this version of transformers_ace.aot_deploy.")
                          .c_str());

  cutoff_ = std::stod(metadata.at("r_max"));
  max_atoms_ = std::stoll(metadata.at("max_atoms"));
  max_edges_ = std::stoll(metadata.at("max_edges"));
  model_type_symbols_ = split_words(metadata.at("type_symbols"));
  const auto z_words = split_words(metadata.at("type_atomic_numbers"));
  if (model_type_symbols_.size() != z_words.size())
    error->all(FLERR, "TRACE AOTI metadata type_symbols/type_atomic_numbers mismatch");
  model_type_atomic_numbers_.clear();
  for (const auto &word : z_words) model_type_atomic_numbers_.push_back(std::stoll(word));

  if (static_cast<int>(model_type_symbols_.size()) != ntypes)
    error->all(FLERR, "TRACE AOTI type map does not match the number of LAMMPS atom types");

  lammps_type_to_z_.assign(ntypes + 1, -1);
  for (int t = 1; t <= ntypes; t++) {
    const std::string symbol(arg[2 + t]);
    auto found = std::find(model_type_symbols_.begin(), model_type_symbols_.end(), symbol);
    if (found == model_type_symbols_.end())
      error->all(FLERR, ("Element " + symbol + " is not in the TRACE model type map").c_str());
    lammps_type_to_z_[t] =
        model_type_atomic_numbers_[std::distance(model_type_symbols_.begin(), found)];
  }

  // Fixed extents: allocate the staging buffers once.
  host_z_.assign(max_atoms_, 1);
  host_pos_.assign(3 * max_atoms_, 0.0f);
  host_mask_.assign(max_atoms_, 0.0f);
  host_edges_.assign(2 * max_edges_, 0);
  allocate_static_inputs();

  if (comm->me == 0) {
    std::cout << "TRACE/AOT: " << model_path << " on " << device_
              << "  r_c=" << cutoff_ << "  capacity: " << max_atoms_ << " atoms, "
              << max_edges_ << " edges\n";
  }

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

double PairTransformersACEAOT::init_one(int, int)
{
  if (!model_loaded_) error->all(FLERR, "pair_style transformers_ace/aot has no model loaded");
  return cutoff_;
}

void PairTransformersACEAOT::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "pair_style transformers_ace/aot requires newton pair on");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairTransformersACEAOT::allocate_static_inputs()
{
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  device_shift_ = torch::zeros({max_edges_, 3}, options);
  device_strain_ = torch::zeros({6}, options);
}

void PairTransformersACEAOT::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Checked before any work, not after: the model returns a single extensive
  // energy and a global virial, so neither per-atom decomposition exists. The
  // per-atom arrays that ev_init just zeroed would otherwise be reported as
  // genuine zeros by compute pe/atom and compute stress/atom.
  if (eflag_atom)
    error->all(FLERR, "pair_style transformers_ace/aot does not provide per-atom energy");
  if (vflag_atom)
    error->all(FLERR, "pair_style transformers_ace/aot does not provide per-atom virial");

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  if (nlocal <= 0) return;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;

  // Serial runs fold ghost forces back onto their owning local atom by tag.
  std::vector<int> force_owner;
  if (comm->nprocs == 1) {
    std::unordered_map<tagint, int> local_by_tag;
    local_by_tag.reserve(nlocal);
    for (int i = 0; i < nlocal; i++) local_by_tag[tag[i]] = i;
    force_owner.assign(nall, -1);
    for (int i = 0; i < nlocal; i++) force_owner[i] = i;
    for (int i = nlocal; i < nall; i++) {
      auto found = local_by_tag.find(tag[i]);
      if (found != local_by_tag.end()) force_owner[i] = found->second;
    }
  }

  if (nall + 2 > max_atoms_) {
    std::string message = "TRACE/AOT: " + std::to_string(nall) +
                          " atoms (local+ghost) exceeds the compiled capacity of " +
                          std::to_string(max_atoms_ - 2) +
                          ". Re-export with a larger --max-atoms.";
    error->one(FLERR, message.c_str());
  }

  // ---- build the directed edge list (neighbour -> centre), as in Eq. (4) ----
  int nedges = 0;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  const double cutsq_model = cutoff_ * cutoff_;

  for (int ii = 0; ii < list->inum; ii++) {
    const int i = ilist[ii];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - x[i][0];
      const double dy = x[j][1] - x[i][1];
      const double dz = x[j][2] - x[i][2];
      if (dx * dx + dy * dy + dz * dz >= cutsq_model) continue;
      if (nedges >= max_edges_) {
        std::string message = "TRACE/AOT: edge count exceeds the compiled capacity of " +
                              std::to_string(max_edges_) + ". Re-export with a larger --max-edges.";
        error->one(FLERR, message.c_str());
      }
      host_edges_[nedges] = j;                 // sender
      host_edges_[max_edges_ + nedges] = i;    // receiver
      nedges++;
    }
  }

  // ---- pad exactly as transformers_ace.aot.pad_lammps_inputs ----
  // Padded atoms: species 1, zero mask. The last padding atom sits at
  // (2 r_c, 0, 0) and the one before it at the origin, so every padded edge has
  // length 2 r_c and the compact envelope gives f_c = 0 identically.
  std::fill(host_z_.begin(), host_z_.end(), 1);
  std::fill(host_pos_.begin(), host_pos_.end(), 0.0f);
  std::fill(host_mask_.begin(), host_mask_.end(), 0.0f);

  for (int i = 0; i < nall; i++) {
    host_z_[i] = lammps_type_to_z_[type[i]];
    host_pos_[3 * i + 0] = static_cast<float>(x[i][0]);
    host_pos_[3 * i + 1] = static_cast<float>(x[i][1]);
    host_pos_[3 * i + 2] = static_cast<float>(x[i][2]);
  }
  // Only owned atoms contribute to the extensive energy; ghosts are their
  // periodic images and would double count.
  for (int i = 0; i < nlocal; i++) host_mask_[i] = 1.0f;
  host_pos_[3 * (max_atoms_ - 1) + 0] = static_cast<float>(2.0 * cutoff_);

  for (int64_t e = nedges; e < max_edges_; e++) {
    host_edges_[e] = max_atoms_ - 1;                 // sender
    host_edges_[max_edges_ + e] = max_atoms_ - 2;    // receiver
  }
  const auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  auto z_t = torch::from_blob(host_z_.data(), {max_atoms_}, i64).to(device_);
  auto pos_t = torch::from_blob(host_pos_.data(), {max_atoms_, 3}, f32).to(device_);
  auto mask_t = torch::from_blob(host_mask_.data(), {max_atoms_}, f32).to(device_);
  auto edge_t = torch::from_blob(host_edges_.data(), {2, max_edges_}, i64).to(device_);
  auto cell_t = cell_tensor().to(device_);

  // The compiled program already contains the derivative graph, so no autograd
  // is needed here. It returns (energy, forces, virial) with virial already in
  // the LAMMPS convention W = -V sigma, including the one-half on shear terms.
  std::vector<at::Tensor> inputs = {z_t,          pos_t,          cell_t, edge_t,
                                    device_shift_, device_strain_, mask_t};
  std::vector<at::Tensor> outputs;
  try {
    outputs = loader_->run(inputs);
  } catch (const std::exception &err) {
    std::string message = std::string("TRACE/AOT model execution failed: ") + err.what();
    error->one(FLERR, message.c_str());
  }
  if (outputs.size() < 3) error->one(FLERR, "TRACE/AOT model returned fewer than three tensors");

  const auto energy = outputs[0].detach().to(torch::kCPU);
  const auto forces = outputs[1].detach().to(torch::kCPU);
  const auto virial_t = outputs[2].detach().to(torch::kCPU);

  auto force_acc = forces.accessor<float, 2>();
  if (comm->nprocs == 1) {
    for (int i = 0; i < nall; i++) {
      const int owner = force_owner[i];
      if (owner < 0) continue;
      f[owner][0] += force_acc[i][0];
      f[owner][1] += force_acc[i][1];
      f[owner][2] += force_acc[i][2];
    }
  } else {
    for (int i = 0; i < nall; i++) {
      f[i][0] += force_acc[i][0];
      f[i][1] += force_acc[i][1];
      f[i][2] += force_acc[i][2];
    }
  }

  eng_vdwl = energy.item<double>();

  if (vflag) {
    auto v = virial_t.accessor<float, 1>();
    for (int k = 0; k < 6; k++) virial[k] += v[k];
  }
}
