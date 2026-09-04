#ifdef PAIR_CLASS

PairStyle(transformers_ace/aot, PairTransformersACEAOT);

#else

#ifndef LMP_PAIR_TRANSFORMERS_ACE_AOT_H
#define LMP_PAIR_TRANSFORMERS_ACE_AOT_H

#include "pair.h"

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

// Ahead-of-time compiled TRACE pair style.
//
// The .pt2 package produced by ``python -m transformers_ace.aot_deploy`` holds a
// fully compiled energy/force/virial program: the derivatives are baked into the
// graph by torch.func.grad, so this pair style performs no autograd of its own.
// The compiled program has *fixed* tensor extents, which is what allows the
// runtime to replay it as a small number of kernel launches instead of the
// ~2300 issued by the eager graph.
//
// Fixed extents are reconciled with LAMMPS' fluctuating atom and neighbour
// counts by padding. Padded atoms carry a zero local-energy mask, and padded
// edges join two padding atoms separated by 2*r_c, so the compact C2 envelope
// makes them contribute exactly zero to the density, the energy, the forces and
// the strain derivative. That inertness is verified to machine precision in
// tests/test_lammps_aot_padding.py.
class PairTransformersACEAOT : public Pair {
 public:
  PairTransformersACEAOT(class LAMMPS *);
  ~PairTransformersACEAOT() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  void allocate();

 private:
  std::map<std::string, std::string> parse_metadata(const std::string &) const;
  std::vector<std::string> split_words(const std::string &) const;
  torch::Tensor cell_tensor() const;
  int local_rank() const;
  torch::Device resolve_device() const;
  void load_package(const std::string &path);

  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
  torch::Device device_ = torch::Device(torch::kCPU);
  std::string device_spec_ = "auto";
  bool model_loaded_ = false;
  double cutoff_ = 0.0;

  // Fixed extents baked into the compiled program.
  int64_t max_atoms_ = 0;
  int64_t max_edges_ = 0;

  std::vector<int64_t> lammps_type_to_z_;
  std::vector<std::string> model_type_symbols_;
  std::vector<int64_t> model_type_atomic_numbers_;

  // Reusable host staging buffers: the compiled program needs the same extents
  // every call, so these are allocated once and refilled in place.
  std::vector<int64_t> host_z_;
  std::vector<float> host_pos_;
  std::vector<float> host_mask_;
  std::vector<int64_t> host_edges_;

  // Two of the seven model inputs are structurally constant for the whole run:
  //
  //   edge_shift  LAMMPS ghosts already carry unwrapped coordinates, so
  //               r_ij = r_j - r_i is the true minimum-image vector and the
  //               lattice shift S_ij is identically zero.
  //   strain      only the virial path perturbs it, and the derivative is
  //               already baked into the compiled program; it is evaluated at
  //               zero strain.
  //
  // At max_edges = 131072 the shift buffer alone is 1.57 MB, which was 41% of
  // the per-step host-to-device traffic and was re-zeroed and re-sent every
  // timestep to no effect. Both live on the device instead, written once.
  torch::Tensor device_shift_;
  torch::Tensor device_strain_;
  void allocate_static_inputs();
};

}    // namespace LAMMPS_NS

#endif
#endif
