#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test images
SMALL_IMAGE="Baboon_256.pgm"
LARGE_IMAGE="cube_1024.pgm"

# Hybrid configurations: MPI_processes x OpenMP_threads (total = 12 cores)
# Format: "processes:threads"
HYBRID_CONFIGS=("1:12" "2:6" "3:4" "4:3" "6:2" "12:1")

# Function to print colored output
print_color() {
    printf "${2}${1}${NC}\n"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_color "Error: File $1 not found!" $RED
        return 1
    fi
    return 0
}

# Function to run serial test
run_serial_test() {
    local image=$1
    local image_info=$(head -3 "$image" | tail -2)
    
    print_color "Testing Serial FFT with $image" $BLUE
    print_color "Image info: $image_info" $YELLOW
    
    # Clean previous outputs
    rm -f fft.pgm ifft.pgm
    
    # Run serial FFT
    echo "Running: ./fft_serial $image"
    time ./fft_serial "$image"
    
    # Check outputs
    if [ -f "fft.pgm" ] && [ -f "ifft.pgm" ]; then
        print_color "✓ Serial test passed - outputs generated" $GREEN
        
        # Check file sizes
        fft_size=$(wc -c < fft.pgm)
        ifft_size=$(wc -c < ifft.pgm)
        print_color "FFT output size: $fft_size bytes" $YELLOW
        print_color "IFFT output size: $ifft_size bytes" $YELLOW
        
        # Move outputs to imgs directory
        mv fft.pgm "imgs/fft_serial_$(basename $image .pgm).pgm"
        mv ifft.pgm "imgs/ifft_serial_$(basename $image .pgm).pgm"
        print_color "✓ Output files moved to imgs/ directory" $GREEN
    else
        print_color "✗ Serial test failed - missing outputs" $RED
        return 1
    fi
    
    echo ""
    return 0
}

# Function to run hybrid test
run_hybrid_test() {
    local image=$1
    local config=$2
    local processes=$(echo $config | cut -d':' -f1)
    local threads=$(echo $config | cut -d':' -f2)
    local image_info=$(head -3 "$image" | tail -2)
    
    print_color "Testing Hybrid MPI+OpenMP FFT with $image using $processes processes x $threads threads" $BLUE
    print_color "Image info: $image_info" $YELLOW
    
    # Clean previous outputs
    rm -f fft.pgm ifft.pgm
    
    # Set OpenMP thread count
    export OMP_NUM_THREADS=$threads
    
    # Run Hybrid FFT
    echo "Running: OMP_NUM_THREADS=$threads mpirun --oversubscribe -np $processes ./fft_hybrid $image"
    time mpirun --oversubscribe -np $processes ./fft_hybrid "$image"
    
    # Check outputs
    if [ -f "fft.pgm" ] && [ -f "ifft.pgm" ]; then
        print_color "✓ Hybrid test passed - outputs generated" $GREEN
        
        # Check file sizes
        fft_size=$(wc -c < fft.pgm)
        ifft_size=$(wc -c < ifft.pgm)
        print_color "FFT output size: $fft_size bytes" $YELLOW
        print_color "IFFT output size: $ifft_size bytes" $YELLOW
        
        # Move outputs to imgs directory
        mv fft.pgm "imgs/fft_hybrid_$(basename $image .pgm)_p${processes}t${threads}.pgm"
        mv ifft.pgm "imgs/ifft_hybrid_$(basename $image .pgm)_p${processes}t${threads}.pgm"
        print_color "✓ Output files moved to imgs/ directory" $GREEN
    else
        print_color "✗ Hybrid test failed - missing outputs" $RED
        return 1
    fi
    
    echo ""
    return 0
}

# Function to run performance comparison
run_performance_test() {
    local image=$1
    print_color "=== Performance Comparison for $image ===" $BLUE
    
    # Serial test
    print_color "Serial Performance:" $YELLOW
    { time ./fft_serial "$image"; } 2>&1 | grep -E "(real|user|sys|Pure)"
    rm -f fft.pgm ifft.pgm
    
    # Hybrid tests
    for config in "${HYBRID_CONFIGS[@]}"; do
        processes=$(echo $config | cut -d':' -f1)
        threads=$(echo $config | cut -d':' -f2)
        print_color "Hybrid Performance (${processes} processes, ${threads} threads):" $YELLOW
        { time OMP_NUM_THREADS=$threads mpirun --oversubscribe -np $processes ./fft_hybrid "$image"; } 2>&1 | grep -E "(real|user|sys|Pure)"
        rm -f fft.pgm ifft.pgm
    done
    echo ""
}

# Function to clean object files
clean_object_files() {
    print_color "Cleaning object files..." $YELLOW
    rm -f *.o
    print_color "✓ Object files cleaned" $GREEN
}

# Function to setup output directory
setup_output_dir() {
    if [ ! -d "imgs" ]; then
        mkdir -p imgs
        print_color "✓ Created imgs directory" $GREEN
    fi
}

# Main script
main() {
    print_color "=== FFT Testing Script ===" $GREEN
    
    # Clean object files at the start
    clean_object_files
    
    # Setup output directory
    setup_output_dir
    
    # Check if executables exist
    if [ "$1" = "serial" ] || [ "$1" = "all" ]; then
        if [ ! -f "./fft_serial" ]; then
            print_color "Error: fft_serial not found. Run 'make fft_serial' first." $RED
            exit 1
        fi
    fi
    
    if [ "$1" = "hybrid" ] || [ "$1" = "all" ]; then
        if [ ! -f "./fft_hybrid" ]; then
            print_color "Error: fft_hybrid not found. Run 'make fft_hybrid' first." $RED
            exit 1
        fi
    fi
    
    # Check if test images exist
    check_file "$SMALL_IMAGE" || exit 1
    check_file "$LARGE_IMAGE" || exit 1
    
    case "$1" in
        "serial")
            print_color "=== Running Serial Tests ===" $GREEN
            run_serial_test "$SMALL_IMAGE"
            run_serial_test "$LARGE_IMAGE"
            ;;
            
        "hybrid")
            print_color "=== Running Hybrid Tests ===" $GREEN
            for config in "${HYBRID_CONFIGS[@]}"; do
                run_hybrid_test "$SMALL_IMAGE" $config
                run_hybrid_test "$LARGE_IMAGE" $config
            done
            ;;
            
        "all"|"")
            print_color "=== Running All Tests ===" $GREEN
            
            # Serial tests
            print_color "--- Serial Tests ---" $YELLOW
            run_serial_test "$SMALL_IMAGE"
            run_serial_test "$LARGE_IMAGE"
            
            # Hybrid tests
            print_color "--- Hybrid Tests ---" $YELLOW
            for config in "${HYBRID_CONFIGS[@]}"; do
                run_hybrid_test "$SMALL_IMAGE" $config
                run_hybrid_test "$LARGE_IMAGE" $config
            done
            
            # Performance comparison
            print_color "--- Performance Comparison ---" $YELLOW
            run_performance_test "$SMALL_IMAGE"
            run_performance_test "$LARGE_IMAGE"
            ;;
            
        "performance")
            print_color "=== Performance Tests Only ===" $GREEN
            run_performance_test "$SMALL_IMAGE"
            run_performance_test "$LARGE_IMAGE"
            ;;
            
        *)
            print_color "Usage: $0 [serial|hybrid|all|performance]" $YELLOW
            print_color "  serial      - Test serial version only" $YELLOW
            print_color "  hybrid      - Test Hybrid MPI+OpenMP version only" $YELLOW
            print_color "  all         - Test both versions (default)" $YELLOW
            print_color "  performance - Run performance comparison only" $YELLOW
            exit 1
            ;;
    esac
    
    print_color "=== Testing Complete ===" $GREEN
    print_color "Generated output files in imgs/ directory:" $YELLOW
    ls -la imgs/*.pgm 2>/dev/null || print_color "No output files found in imgs/" $RED
    
    # Clean object files at the end as well
    clean_object_files
}

# Make script executable and run
main "$@"