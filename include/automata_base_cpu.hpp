#ifndef AUTOMATA_HPP_
#define AUTOMATA_HPP_

#include "automata_interface.hpp"
#include "config.hpp"
#include "grid.hpp"
#include "types.hpp"

namespace cpu {

class AutomataBase : public AutomataInterface {
  public:
    AutomataBase(const unsigned long pRandSeed,
                 std::ostringstream *const pLiveLogBuffer,
                 void (*pUpdateBuffersFunc)());
    virtual ~AutomataBase();
    virtual void compute_grid(const bool logEnabled = false) override;
    virtual void update_grid_buffers() override { mUpdateBuffersFunc(); }

  protected:
    uint mActiveCellCount;
    std::ostringstream *mLiveLogBuffer;

    bool compute_cell(uint i, uint j);

  private:
    void (*mUpdateBuffersFunc)();
};

} // namespace cpu

#endif