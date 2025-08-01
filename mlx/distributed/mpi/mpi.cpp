// Copyright © 2024 Apple Inc.

#include <dlfcn.h>
#include <iostream>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/mpi/mpi.h"
#include "mlx/distributed/mpi/mpi_declarations.h"

#define LOAD_SYMBOL(symbol, variable)                              \
  {                                                                \
    variable = (decltype(variable))dlsym(libmpi_handle_, #symbol); \
    char* error = dlerror();                                       \
    if (error != nullptr) {                                        \
      libmpi_handle_ = nullptr;                                    \
      return;                                                      \
    }                                                              \
  }

#ifdef __APPLE__
static constexpr const char* libmpi_name = "libmpi.dylib";
#else
static constexpr const char* libmpi_name = "libmpi.so";
#endif

namespace mlx::core::distributed::mpi {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

namespace {

template <typename T>
void simple_sum(
    void* input,
    void* accumulator,
    int* len,
    MPI_Datatype* datatype) {
  T* in = (T*)input;
  T* acc = (T*)accumulator;
  int N = *len;

  while (N-- > 0) {
    *acc += *in;
    acc++;
    in++;
  }
}
template void simple_sum<float16_t>(void*, void*, int*, MPI_Datatype*);
template void simple_sum<bfloat16_t>(void*, void*, int*, MPI_Datatype*);

template <typename T>
void simple_max(
    void* input,
    void* accumulator,
    int* len,
    MPI_Datatype* datatype) {
  T* in = (T*)input;
  T* acc = (T*)accumulator;
  int N = *len;

  while (N-- > 0) {
    *acc = std::max(*acc, *in);
    acc++;
    in++;
  }
}
template void simple_max<float16_t>(void*, void*, int*, MPI_Datatype*);
template void simple_max<bfloat16_t>(void*, void*, int*, MPI_Datatype*);
template void simple_max<complex64_t>(void*, void*, int*, MPI_Datatype*);

template <typename T>
void simple_min(
    void* input,
    void* accumulator,
    int* len,
    MPI_Datatype* datatype) {
  T* in = (T*)input;
  T* acc = (T*)accumulator;
  int N = *len;

  while (N-- > 0) {
    *acc = std::min(*acc, *in);
    acc++;
    in++;
  }
}
template void simple_min<float16_t>(void*, void*, int*, MPI_Datatype*);
template void simple_min<bfloat16_t>(void*, void*, int*, MPI_Datatype*);
template void simple_min<complex64_t>(void*, void*, int*, MPI_Datatype*);

struct MPIWrapper {
  MPIWrapper() {
    initialized_ = false;

    libmpi_handle_ = dlopen(libmpi_name, RTLD_NOW | RTLD_GLOBAL);
    if (libmpi_handle_ == nullptr) {
      return;
    }

    // Check library version and warn if it isn't Open MPI
    int (*get_version)(char*, int*);
    LOAD_SYMBOL(MPI_Get_library_version, get_version);
    char version_ptr[MPI_MAX_LIBRARY_VERSION_STRING];
    int version_length = 0;
    get_version(version_ptr, &version_length);
    std::string_view version(version_ptr, version_length);
    if (version.find("Open MPI") == std::string::npos) {
      std::cerr << "[mpi] MPI found but it does not appear to be Open MPI."
                << "MLX requires Open MPI but this is " << version << std::endl;
      libmpi_handle_ = nullptr;
      return;
    }

    // API
    LOAD_SYMBOL(MPI_Init, init);
    LOAD_SYMBOL(MPI_Finalize, finalize);
    LOAD_SYMBOL(MPI_Comm_rank, rank);
    LOAD_SYMBOL(MPI_Comm_size, size);
    LOAD_SYMBOL(MPI_Comm_split, comm_split);
    LOAD_SYMBOL(MPI_Comm_free, comm_free);
    LOAD_SYMBOL(MPI_Allreduce, all_reduce);
    LOAD_SYMBOL(MPI_Allgather, all_gather);
    LOAD_SYMBOL(MPI_Send, send);
    LOAD_SYMBOL(MPI_Recv, recv);
    LOAD_SYMBOL(MPI_Type_contiguous, mpi_type_contiguous);
    LOAD_SYMBOL(MPI_Type_commit, mpi_type_commit);
    LOAD_SYMBOL(MPI_Op_create, mpi_op_create);

    // Objects
    LOAD_SYMBOL(ompi_mpi_comm_world, comm_world_);

    // Ops
    LOAD_SYMBOL(ompi_mpi_op_sum, op_sum_);
    LOAD_SYMBOL(ompi_mpi_op_max, op_max_);
    LOAD_SYMBOL(ompi_mpi_op_min, op_min_);

    // Datatypes
    LOAD_SYMBOL(ompi_mpi_c_bool, mpi_bool_);
    LOAD_SYMBOL(ompi_mpi_int8_t, mpi_int8_);
    LOAD_SYMBOL(ompi_mpi_uint8_t, mpi_uint8_);
    LOAD_SYMBOL(ompi_mpi_int16_t, mpi_int16_);
    LOAD_SYMBOL(ompi_mpi_uint16_t, mpi_uint16_);
    LOAD_SYMBOL(ompi_mpi_int32_t, mpi_int32_);
    LOAD_SYMBOL(ompi_mpi_uint32_t, mpi_uint32_);
    LOAD_SYMBOL(ompi_mpi_int64_t, mpi_int64_);
    LOAD_SYMBOL(ompi_mpi_uint64_t, mpi_uint64_);
    LOAD_SYMBOL(ompi_mpi_float, mpi_float_);
    LOAD_SYMBOL(ompi_mpi_double, mpi_double_);
    LOAD_SYMBOL(ompi_mpi_c_complex, mpi_complex_);
  }

  bool is_available() {
    return libmpi_handle_ != nullptr;
  }

  bool init_safe() {
    if (!is_available()) {
      return false;
    }
    bool success = init(nullptr, nullptr) == MPI_SUCCESS;

    // Initialize custom types and ops
    if (success && !initialized_) {
      // Custom float16 dtypes
      mpi_type_contiguous(2, mpi_uint8_, &mpi_float16_);
      mpi_type_commit(&mpi_float16_);
      mpi_type_contiguous(2, mpi_uint8_, &mpi_bfloat16_);
      mpi_type_commit(&mpi_bfloat16_);

      // Custom reduction ops
      mpi_op_create(&simple_sum<float16_t>, 1, &op_sum_f16_);
      mpi_op_create(&simple_sum<bfloat16_t>, 1, &op_sum_bf16_);
      mpi_op_create(&simple_max<float16_t>, 1, &op_max_f16_);
      mpi_op_create(&simple_max<bfloat16_t>, 1, &op_max_bf16_);
      mpi_op_create(&simple_max<complex64_t>, 1, &op_max_c64_);
      mpi_op_create(&simple_min<float16_t>, 1, &op_min_f16_);
      mpi_op_create(&simple_min<bfloat16_t>, 1, &op_min_bf16_);
      mpi_op_create(&simple_min<complex64_t>, 1, &op_min_c64_);

      initialized_ = true;
    }

    return success;
  }

  void finalize_safe() {
    if (is_available()) {
      finalize();
    }
  }

  MPI_Comm world() {
    return comm_world_;
  }

  MPI_Datatype datatype(const array& arr) {
    switch (arr.dtype()) {
      case bool_:
        return mpi_bool_;
      case int8:
        return mpi_int8_;
      case uint8:
        return mpi_uint8_;
      case int16:
        return mpi_int16_;
      case uint16:
        return mpi_uint16_;
      case int32:
        return mpi_int32_;
      case uint32:
        return mpi_uint32_;
      case int64:
        return mpi_int64_;
      case uint64:
        return mpi_uint64_;
      case float32:
        return mpi_float_;
      case complex64:
        return mpi_complex_;
      case float16:
        return mpi_float16_;
      case bfloat16:
        return mpi_bfloat16_;
      case float64:
        return mpi_double_;
      default:
        throw std::runtime_error("Invalid type");
    }
  }

  MPI_Op op_sum(const array& arr) {
    switch (arr.dtype()) {
      case float16:
        return op_sum_f16_;
      case bfloat16:
        return op_sum_bf16_;
      default:
        return op_sum_;
    }
  }

  MPI_Op op_max(const array& arr) {
    switch (arr.dtype()) {
      case float16:
        return op_max_f16_;
      case bfloat16:
        return op_max_bf16_;
      case complex64:
        return op_max_c64_;
      default:
        return op_max_;
    }
  }

  MPI_Op op_min(const array& arr) {
    switch (arr.dtype()) {
      case float16:
        return op_min_f16_;
      case bfloat16:
        return op_min_bf16_;
      case complex64:
        return op_min_c64_;
      default:
        return op_min_;
    }
  }

  void* libmpi_handle_;

  // API
  int (*init)(int*, char***);
  int (*finalize)();
  int (*rank)(MPI_Comm, int*);
  int (*size)(MPI_Comm, int*);
  int (*all_reduce)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
  int (*all_gather)(
      const void*,
      int,
      MPI_Datatype,
      void*,
      int,
      MPI_Datatype,
      MPI_Comm);
  int (*comm_split)(MPI_Comm, int, int, MPI_Comm*);
  int (*comm_free)(MPI_Comm*);
  int (*send)(const void*, int, MPI_Datatype, int, int, MPI_Comm);
  int (*recv)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*);

  // Objects
  MPI_Comm comm_world_;

  // Ops
  MPI_Op op_sum_;
  MPI_Op op_sum_f16_;
  MPI_Op op_sum_bf16_;
  MPI_Op op_max_;
  MPI_Op op_max_f16_;
  MPI_Op op_max_bf16_;
  MPI_Op op_max_c64_;
  MPI_Op op_min_;
  MPI_Op op_min_f16_;
  MPI_Op op_min_bf16_;
  MPI_Op op_min_c64_;

  // Datatypes
  MPI_Datatype mpi_bool_;
  MPI_Datatype mpi_int8_;
  MPI_Datatype mpi_uint8_;
  MPI_Datatype mpi_int16_;
  MPI_Datatype mpi_uint16_;
  MPI_Datatype mpi_int32_;
  MPI_Datatype mpi_uint32_;
  MPI_Datatype mpi_int64_;
  MPI_Datatype mpi_uint64_;
  MPI_Datatype mpi_float_;
  MPI_Datatype mpi_double_;
  MPI_Datatype mpi_complex_;
  MPI_Datatype mpi_float16_;
  MPI_Datatype mpi_bfloat16_;

 private:
  bool initialized_;

  // Private API
  int (*mpi_type_contiguous)(int, MPI_Datatype, MPI_Datatype*);
  int (*mpi_type_commit)(MPI_Datatype*);
  int (*mpi_op_create)(MPI_User_function*, int, MPI_Op*);
};

MPIWrapper& mpi() {
  static MPIWrapper wrapper;
  return wrapper;
}

} // namespace

class MPIGroup : public GroupImpl {
 public:
  MPIGroup(MPI_Comm comm, bool global)
      : comm_(comm), global_(global), rank_(-1), size_(-1) {}

  virtual ~MPIGroup() {
    if (global_) {
      mpi().finalize_safe();
    } else {
      mpi().comm_free(&comm_);
    }
  }

  int rank() override {
    if (rank_ < 0) {
      mpi().rank(comm_, &rank_);
    }
    return rank_;
  }

  int size() override {
    if (size_ < 0) {
      mpi().size(comm_, &size_);
    }
    return size_;
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    key = (key < 0) ? rank() : key;

    MPI_Comm new_comm;
    int result = mpi().comm_split(comm_, color, key, &new_comm);
    if (result != MPI_SUCCESS) {
      throw std::runtime_error("MPI could not split this group");
    }

    return std::make_shared<MPIGroup>(new_comm, false);
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch(
        mpi().all_reduce,
        (input.data<void>() == output.data<void>()) ? MPI_IN_PLACE
                                                    : input.data<void>(),
        output.data<void>(),
        input.size(),
        mpi().datatype(input),
        mpi().op_sum(input),
        comm_);
  }

  void all_max(const array& input, array& output, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch(
        mpi().all_reduce,
        (input.data<void>() == output.data<void>()) ? MPI_IN_PLACE
                                                    : input.data<void>(),
        output.data<void>(),
        input.size(),
        mpi().datatype(input),
        mpi().op_max(input),
        comm_);
  }

  void all_min(const array& input, array& output, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch(
        mpi().all_reduce,
        (input.data<void>() == output.data<void>()) ? MPI_IN_PLACE
                                                    : input.data<void>(),
        output.data<void>(),
        input.size(),
        mpi().datatype(input),
        mpi().op_min(input),
        comm_);
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch(
        mpi().all_gather,
        input.data<void>(),
        input.size(),
        mpi().datatype(input),
        output.data<void>(),
        input.size(),
        mpi().datatype(output),
        comm_);
  }

  void send(const array& input, int dst, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.dispatch(
        mpi().send,
        input.data<void>(),
        input.size(),
        mpi().datatype(input),
        dst,
        0,
        comm_);
  }

  void recv(array& out, int src, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_output_array(out);
    encoder.dispatch([out_ptr = out.data<void>(),
                      out_size = out.size(),
                      out_type = mpi().datatype(out),
                      src,
                      comm = comm_]() {
      MPI_Status status;
      mpi().recv(out_ptr, out_size, out_type, src, MPI_ANY_TAG, comm, &status);
    });
  }

 private:
  MPI_Comm comm_;
  bool global_;
  int rank_;
  int size_;
};

bool is_available() {
  return mpi().is_available();
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (!mpi().init_safe()) {
    if (strict) {
      throw std::runtime_error("Cannot initialize MPI");
    }
    return nullptr;
  }

  return std::make_shared<MPIGroup>(mpi().world(), true);
}

} // namespace mlx::core::distributed::mpi
