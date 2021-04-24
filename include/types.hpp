#ifndef TYPES_HPP_
#define TYPES_HPP_

typedef unsigned int uint;

// A bidimensional unsigned int vector
typedef struct suvec2 {
    uint x, y;

    suvec2(){}; // no initialization
    suvec2(uint _x, uint _y) : x(_x), y(_y){};
} uvec2;

// A bidimensional float vector
typedef struct sfvec2 {
    float x, y;

    sfvec2(){}; // no initialization
    sfvec2(float _x, float _y) : x(_x), y(_y){};
} fvec2;

// A bidimensional float vector with integer state
typedef struct sfvec2s {
    fvec2 position;
    int state;

    sfvec2s(){}; // no initialization
    sfvec2s(fvec2 _p, int _s) : position(_p), state(_s){};
} fvec2s;

// Grid render info including cell density information and number of vertices
typedef struct sGridRenderInfo {
    uint totalVertices;
    fvec2 numVertices;
    fvec2 cellDensity;

    sGridRenderInfo(){}; // no initialization
    sGridRenderInfo(fvec2 _n, fvec2 _c) : numVertices(_n), cellDensity(_c) {
        totalVertices = _n.x * _n.y;
    };
    sGridRenderInfo(uint _t, fvec2 _n, fvec2 _c)
        : totalVertices(_t), numVertices(_n), cellDensity(_c){};
} GridRenderInfo;

#endif