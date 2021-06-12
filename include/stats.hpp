namespace stats {

extern unsigned long iterations;
// time in nanoseconds
extern unsigned long loadTime;
extern unsigned long totalEvolveTime;
extern unsigned long totalBufferTime;

/* Outputs:
 * iterations, total load time, total evolve time,
 * average evolve time, average render time */
void print_timings();

}; // namespace stats