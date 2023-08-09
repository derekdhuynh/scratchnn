#include "Autograd.h"

namespace scratchnn {

// Create the global execution engine
static auto runtime_engine = std::make_shared<ExecutionEngine>();

} // namespace scratchnn
