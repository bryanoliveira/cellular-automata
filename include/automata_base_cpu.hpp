#ifndef AUTOMATA_HPP_
#define AUTOMATA_HPP_

#include "automata_interface.hpp"
#include "config.hpp"
#include "grid.hpp"

namespace cpu {

class AutomataBase : public AutomataInterface {
  public:
    AutomataBase(unsigned long pRandSeed,
                 std::ostringstream *const pLiveLogBuffer,
                 void (*pUpdateBuffersFunc)());
    virtual ~AutomataBase();
    virtual void compute_grid(bool logEnabled = false);
    virtual void update_grid_buffers() { mUpdateBuffersFunc(); }

  protected:
    unsigned int mActiveCellCount;
    std::ostringstream *mLiveLogBuffer;

    bool compute_cell(unsigned int i, unsigned int j);

  private:
    void (*mUpdateBuffersFunc)();
};

} // namespace cpu

#endif