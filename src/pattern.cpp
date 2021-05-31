#include <chrono>
#include <fstream>
#include <sstream>
#include <spdlog/spdlog.h>

#include "pattern.hpp"
#include "stats.hpp"

#define STATE_INIT 0
#define STATE_COMMENT 1
#define STATE_HEADER_X 2
#define STATE_HEADER_Y 3
#define STATE_RULE 4
#define STATE_PATTERN 5
#define STATE_END 6

void load_pattern(const std::string filename) {
    spdlog::info("Loading initial pattern...");
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    // pattern size
    unsigned int sizeCols, sizeRows;
    // pattern start position
    unsigned int startCol;
    // row/col controllers
    unsigned int row, col;

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        spdlog::critical("Rule load error: " + filename);
        exit(EXIT_FAILURE);
    }
    std::stringstream buffer;
    char ch;

    unsigned short state = STATE_INIT;

    while (infile.get(ch)) {
        // an automata to initialize an automata
        switch (state) {
        case STATE_INIT:
            // line init can be a comment or the file header
            if (ch == '#')
                state = STATE_COMMENT;
            else if (ch == 'x')
                state = STATE_HEADER_X;
            break;
        case STATE_COMMENT:
            // ignore all chars, except for endl
            if (ch == '\n')
                state = STATE_INIT;
            break;
        case STATE_HEADER_X:
            // after we read the char "x" we skip the " = "
            if (ch == ' ' || ch == '=')
                continue;
            // if we reached ",", process the buffer into a number
            else if (ch == ',') {
                buffer >> sizeCols;
                buffer.clear();
                // calculate the col start position (integer) and save it for
                // later
                col = startCol = config::cols / 2 - sizeCols / 2;
                // the next number we'll read is Y
                state = STATE_HEADER_Y;
            }
            // read the number
            else if (ch >= '0' && ch <= '9')
                buffer << ch;
            break;
        case STATE_HEADER_Y:
            // skip "y", "=" and " "
            if (ch == 'y' || ch == '=' || ch == ' ')
                continue;
            // if we reached ",", process the buffer into a number
            else if (ch == ',') {
                buffer >> sizeRows;
                buffer.clear();
                // calculate the row start position (integer)
                row = config::rows / 2 - sizeRows / 2;
                spdlog::info("Pattern size: {}x{}", sizeRows, sizeCols);
                spdlog::info("Start position: {}x{}", row, startCol);
                // the next thing is the rule
                state = STATE_RULE;
            }
            // read the number
            else if (ch >= '0' && ch <= '9')
                buffer << ch;
            break;
        case STATE_RULE:
            // we don't read the rules yet
            if (ch == '\n')
                state = STATE_PATTERN;
            break;
        case STATE_PATTERN:
            // put numbers into buffer
            if (ch >= '0' && ch <= '9')
                buffer << ch;
            // parse the current buffer
            else if (ch == 'o' || ch == 'b' || ch == '$') {
                // define segment length
                unsigned int len = 1;
                // check if a number was fed
                if (buffer.rdbuf()->in_avail() > 0) {
                    buffer >> len;
                    buffer.str(""); // clear chars
                    buffer.clear();
                }

                // if the char is "alive" (o) fill in the grid
                if (ch == 'o') {
                    fill_grid(row, col, len);
                    col += len; // update cursor
                }
                // if the char is "dead" (b) no operation needs to be done
                else if (ch == 'b') {
                    // just update the cursor
                    col += len;
                }
                // if the char is "new line" ($) skip the number of rows set
                else if (ch == '$') {
                    row += len;     // skip rows
                    col = startCol; // reset col
                }
            } else if (ch == '!')
                // end the read process
                state = STATE_END;
            break;
        }
    }

    stats::loadTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - timeStart)
                          .count();
    spdlog::info("Pattern loading done.");
}

void fill_grid(const uint row, const uint col, const uint length) {
    // enables a contiguous segment of the grid
    for (uint i = col; i < col + length; i++) {
        grid[row * config::cols + i] = true;
    }
}