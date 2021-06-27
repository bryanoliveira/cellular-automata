#ifndef AUTOMATA_INTERFACE_HPP_
#define AUTOMATA_INTERFACE_HPP_

#include "types.hpp"

class AutomataInterface {
  public:
    virtual ~AutomataInterface() {}
    virtual void prepare() = 0;
    virtual void evolve(bool logEnabled) = 0;
    virtual void update_grid_buffers() = 0;
};

#endif