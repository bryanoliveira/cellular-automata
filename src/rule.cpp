#include <iostream>
#include <fstream>
#include <sstream>

#include "rule.hpp"

#define STATE_INIT 0
#define STATE_COMMENT 1
#define STATE_HEADER_X 2
#define STATE_HEADER_Y 3
#define STATE_RULE 4
#define STATE_GRID 5
#define STATE_END 6

void load_rule(std::string filename) {
    // pattern size
    unsigned int sizeCols, sizeRows;
    // pattern start position
    unsigned int startCol;
    // row/col controllers
    unsigned int row, col;

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        fprintf(stderr, "Rule load error: %s\n", filename.c_str());
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
            else
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
                std::cout << "Pattern size: " << sizeRows << "x" << sizeCols
                          << std::endl;
                std::cout << "Start position: " << row << "x" << startCol
                          << std::endl;
                // the next thing is the rule
                state = STATE_RULE;
            }
            // read the number
            else
                buffer << ch;
            break;
        case STATE_RULE:
            // we don't read the rules yet
            if (ch == '\n')
                state = STATE_GRID;
            break;
        case STATE_GRID:
            // put numbers into buffer
            if (ch >= '0' && ch <= '9')
                buffer << ch;
            // parse the current buffer
            else if (ch == 'o' || ch == 'b') {
                // define segment length
                unsigned int len = 1;
                // check if a number was fed
                if (buffer.rdbuf()->in_avail() > 0) {
                    buffer >> len;
                    buffer.clear();
                }
                // if the char is "alive" fill in the grid
                if (ch == 'o')
                    fill_grid(row, col, len);
                // update cursor
                col += len;
            } else if (ch == '$') {
                // move cursor to the next grid row
                row++;
                col = startCol;
            } else if (ch == '!')
                // end the read process
                state = STATE_END;
            break;
        }
    }
}

void fill_grid(unsigned int row, unsigned int col, unsigned int length) {
    // std::cout << "Inserting " << length << " cells from " << row << "x" <<
    // col << std::endl;
    for (unsigned int i = col; i < col + length; i++) {
        grid[row * config::cols + i] = true;
    }
}