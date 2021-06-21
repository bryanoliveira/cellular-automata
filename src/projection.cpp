#ifndef HEADLESS_ONLY

#include <cmath>

#include "projection.hpp"

namespace proj {

// - static values
GridRenderInfo info;

// - dynamic values
// default density is 1:1
uvec2 cellDensity(1, 1);
ulim2 gridLimX, gridLimY;

// - private members

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *const delta, const ulim2 ref,
                      const int sectionSize, const int hardLimit);

void init() {
    controls::minScale = 1;

    // define sizes & proportions
    if (config::noDownsample) {
        info.numVertices = {config::cols, config::rows};
        info.totalVertices = config::cols * config::rows;
        info.cellDensity = {1, 1};

        controls::maxScale = 100.0f;
        controls::scaleFactor = 0.99f; // related to render space

        controls::translateFactor = 0.002f; // related to render space
    } else {
        info.numVertices.x =
            config::cols > config::width ? config::width : config::cols;
        info.numVertices.y =
            config::rows > config::height ? config::height : config::rows;
        info.totalVertices = info.numVertices.x * info.numVertices.y;
        // max density
        info.cellDensity.x = config::cols / info.numVertices.x;
        info.cellDensity.y = config::rows / info.numVertices.y;

        // max scale will give us 1:1 cell to vertice mapping
        controls::maxScale = std::max(info.cellDensity.x, info.cellDensity.y);
        // scale factor should give us density steps of 1
        controls::scaleFactor = 1;

        // related to cell space
        controls::translateFactor =
            static_cast<float>(config::cols) / info.numVertices.x;
    }

    // define default gridEnd (which is defined at run-time)
    gridLimX = {0, config::cols};
    gridLimY = {0, config::rows};
}

void update() {
    // if downsampling is disabled, there's nothing to do
    if (config::noDownsample)
        return;

    // how many grid cells will be mapped to each vertice
    cellDensity = {
        uint(std::max(config::cols / info.numVertices.x - controls::scale,
                      1.0f)),
        uint(std::max(config::rows / info.numVertices.y - controls::scale,
                      1.0f))};

    // section size will start with the whole grid (scale = 1)
    const uvec2 sectionSize(info.numVertices.x * cellDensity.x,
                            info.numVertices.y * cellDensity.y);
    // the indices of the considered grid sections
    const ulim2 refX((config::cols / 2.0) - sectionSize.x / 2,
                     (config::cols / 2.0) + sectionSize.x / 2);
    const ulim2 refY((config::rows / 2.0) - sectionSize.y / 2,
                     (config::rows / 2.0) + sectionSize.y / 2);

    // calculate translations
    gridLimX = translateLimits(&controls::position.x, refX, sectionSize.x,
                               config::cols);
    gridLimY = translateLimits(&controls::position.y, refY, sectionSize.y,
                               config::rows);
}

// This modifies the param delta in order to limit it!
ulim2 translateLimits(float *const delta, const ulim2 ref,
                      const int sectionSize, const int hardLimit) {
    const ulim2 translated = {
        uint(std::min(std::max(static_cast<int>(ref.start + *delta), 0),
                      hardLimit)),
        uint(std::max(std::min(static_cast<int>(ref.end + *delta), hardLimit),
                      sectionSize)),
    };

    // crop delta by the amount used
    if (*delta < 0)
        // this may be negative - convert it before the operation
        *delta =
            static_cast<int>(translated.start) - static_cast<int>(ref.start);
    else if (*delta > 0)
        // this will be always positive
        *delta = translated.end - ref.end;

    return translated;
}

} // namespace proj

#endif // HEADLESS_ONLY