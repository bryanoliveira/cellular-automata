#ifndef TYPES_HPP_
#define TYPES_HPP_

typedef unsigned int uint;

// A bidimensional int limit object
// It is essentially the same as a vec2 but with different member names for
// semantic predictability
typedef struct slim2 {
    int start, end;

    slim2(){}; // no initialization
    slim2(int _s, int _e) : start(_s), end(_e){};
} lim2;

// A bidimensional unsigned int limit object
// It is essentially the same as a uvec2 but with different member names for
// semantic predictability
typedef struct sulim2 {
    uint start, end;

    sulim2(){}; // no initialization
    sulim2(uint _s, uint _e) : start(_s), end(_e){};
    uint range() { return end - start; };
} ulim2;

// A bidimensional int vector
typedef struct svec2 {
    int x, y;

    svec2(){}; // no initialization
    svec2(int _x, int _y) : x(_x), y(_y){};
} vec2;

// A bidimensional unsigned int vector
// It is essentially the same as a ulim2 but with different member names for
// semantic predictability
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
    uvec2 numVertices;
    fvec2 cellDensity;

    sGridRenderInfo(){}; // no initialization
    sGridRenderInfo(uvec2 _n, fvec2 _c) : numVertices(_n), cellDensity(_c) {
        totalVertices = _n.x * _n.y;
    };
    sGridRenderInfo(uint _t, uvec2 _n, fvec2 _c)
        : totalVertices(_t), numVertices(_n), cellDensity(_c){};
} GridRenderInfo;

#endif