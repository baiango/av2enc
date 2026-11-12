#include "compound_full_cpu_oracle.h"

namespace av2enc::cuda::compound_full::test {

void run_compound_job_cpu(const av2enc::inter::adv::CompoundJob& job) {
  av2enc::inter::adv::run_compound_job(job);
}

}  // namespace av2enc::cuda::compound_full::test
