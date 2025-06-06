# Compiler settings
CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O2 -lm
HYBRID_CFLAGS = -Wall -O2 -lm -fopenmp

# Source files
SERIAL_SOURCES = fft_serial.c pgm.c cshift.c
HYBRID_SOURCES = fft_hybrid.c pgm.c cshift.c
HEADERS = pgm.h cshift.h

# Target executables
SERIAL_TARGET = fft_serial
HYBRID_TARGET = fft_hybrid

# Object files
SERIAL_OBJECTS = $(SERIAL_SOURCES:.c=.o)
HYBRID_OBJECTS = $(HYBRID_SOURCES:.c=_hybrid.o)

# Default target
all: $(SERIAL_TARGET) $(HYBRID_TARGET)

# Serial version
$(SERIAL_TARGET): $(SERIAL_OBJECTS)
	$(CC) $(SERIAL_OBJECTS) -o $(SERIAL_TARGET) $(CFLAGS)
	@echo "Serial FFT compiled successfully!"

# Hybrid MPI+OpenMP version
$(HYBRID_TARGET): $(HYBRID_OBJECTS)
	$(MPICC) $(HYBRID_OBJECTS) -o $(HYBRID_TARGET) $(HYBRID_CFLAGS)
	@echo "Hybrid MPI+OpenMP FFT compiled successfully!"

# Serial object files
%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@ $(CFLAGS)

# Hybrid object files (to avoid conflicts)
%_hybrid.o: %.c $(HEADERS)
	$(MPICC) -c $< -o $@ $(HYBRID_CFLAGS)

# Clean build files
clean:
	rm -f *.o $(SERIAL_TARGET) $(HYBRID_TARGET)
	rm -f fft.pgm ifft.pgm
	@echo "Cleaned all build files and output images"

# Clean only object files
clean-objects:
	rm -f *.o
	@echo "Cleaned object files"

# Clean only output images
clean-output:
	rm -f fft.pgm ifft.pgm
	@echo "Cleaned output images"

# Organize existing pgm files into imgs directory
organize-imgs:
	@mkdir -p imgs
	@if ls *.pgm 1> /dev/null 2>&1; then \
		mv *.pgm imgs/ && echo "Moved existing .pgm files to imgs/ directory"; \
	else \
		echo "No .pgm files found to organize"; \
	fi

# Test targets
test-serial: $(SERIAL_TARGET)
	@echo "Running serial tests..."
	@./test_fft.sh serial

test-hybrid: $(HYBRID_TARGET)
	@echo "Running hybrid tests..."
	@./test_fft.sh hybrid

test-all: $(SERIAL_TARGET) $(HYBRID_TARGET)
	@echo "Running all tests..."
	@./test_fft.sh all

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build serial and hybrid versions"
	@echo "  fft_serial  - Build serial version only"
	@echo "  fft_hybrid  - Build hybrid MPI+OpenMP version only"
	@echo "  clean       - Remove all build files and output images"
	@echo "  clean-objects- Remove only object files"
	@echo "  clean-output- Remove only output images"
	@echo "  organize-imgs- Move existing .pgm files to imgs/ directory"
	@echo "  test-serial - Run serial tests"
	@echo "  test-hybrid - Run hybrid tests"
	@echo "  test-all    - Run all tests"
	@echo "  help        - Show this help message"

# Phony targets
.PHONY: all clean clean-objects clean-output organize-imgs test-serial test-hybrid test-all help