#ifndef STATS_HPP_
#define STATS_HPP_

namespace stats {

extern ulong iterations;
// time in nanoseconds
extern ulong loadTime;
extern ulong totalEvolveTime;
extern ulong totalBufferTime;

/* Outputs:
 * iterations, total load time, total evolve time,
 * average evolve time, average render time */
void print_timings();

}; // namespace stats

#endif // STATS_HPP_