#include <iostream>
#include <fstream>
#include <sstream>

#define STATE_INIT 0
#define STATE_HEADER_X 2
#define STATE_HEADER_Y 3
#define STATE_RULE 4
#define STATE_GRID 5
#define STATE_END 6

void load_rule(unsigned int posX, unsigned int posY) {
    unsigned long x, y;

    std::ifstream infile("rules/glider.rle");
    std::string line;

    unsigned short state = STATE_INIT;

    while (std::getline(infile, line)) {
        std::cout << "buf: " << line << std::endl;

        // skip comments
        if (line[0] == '#')
            continue;

        std::istringstream buffer(line);
        std::string token;
        while (buffer >> token) {
            std::cout << "token: " << token << " | state: " << state
                      << std::endl;
            bool skipLine = false;

            switch (state) {
            case STATE_INIT:
                // since we skip all comments, the file starts with the header
                // and sets the X value first
                if (token == "x")
                    state = STATE_HEADER_X;
                break;
            case STATE_HEADER_X:
                // after we read the char "x" we skip the "=" and read the
                // number without ",", which is the last char
                if (token != "=") {
                    x = std::stoul(token.substr(0, token.size() - 1));
                    // the next number we'll read is Y
                    state = STATE_HEADER_Y;
                }
                break;
            case STATE_HEADER_Y:
                // we skip "y" and "=" and read the number without ","
                if (token != "y" && token != "=") {
                    y = std::stoul(token.substr(0, token.size() - 1));
                    state = STATE_RULE;
                }
                break;
            case STATE_RULE:
                // we don't read the rules yet
                skipLine = true;
                state = STATE_GRID;
                break;
            // TODO perhaps read the file char by char?
            case STATE_GRID:
                // iterate over the token chars
                bool alive;
                unsigned long repeats = 1;
                std::ostringstream repeatStream;
                for (int i = 0; i < token.size(); i++) {
                    if (token[i] >= '0' && token[i] <= '9')
                        repeatStream << token[i];
                    else if (token[i] == 'o')
                        alive = true;
                    else if (token[i] == 'b')
                        alive = false;
                    else if (token[i] == '$') {
                        //  TODO add cell states to grid & reset buffers
                    }
                }
                break;
            }

            // skip to next line
            if (skipLine)
                break;
        }
    }

    std::cout << "x: " << x << " y: " << y << std::endl;
}