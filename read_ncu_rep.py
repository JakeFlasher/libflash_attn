import sys
import os
import csv
import glob
import argparse
from ncu_report_helper import *
def find_nsight_folder(cuda_base_path):
    for root, dirs, files in os.walk(cuda_base_path):
        if "nsight-compute-" in root and "extras/python" in root:
            return root
    return None

def detect_cuda_path():
    cuda_base_path = "/usr/local/cuda-"
    cuda_versions = ["11.0", "11.8", "12.0", "12.1", "12.3"]

    for version in cuda_versions:
        cuda_path = cuda_base_path + version
        result = find_nsight_folder(cuda_path)
        if result:
            print(f"\033[0;32;40mUsing ncu_report.py module in path: {result}\033[0m")
            return result
    
    print(f"\033[0;31;40mNo compatible CUDA version found with ncu_report.py.\033[0m")
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract metrics from NCU reports to CSV')
    parser.add_argument('--reports-dir', type=str, default='./ncu_reports', 
                      help='Root directory containing subdirectories with NCU report files')
    parser.add_argument('--cuda-path', type=str, default=None,
                      help='Path to CUDA installation (optional, will auto-detect if not provided)')
    return parser.parse_args()

def process_subdirectory(subdir_path, ncu_report_module):
    """Process all NCU reports in a subdirectory and create a CSV file in that subdirectory"""
    print(f"Processing subdirectory: {subdir_path}")
    
    # Find all NCU report files in the subdirectory
    report_files = glob.glob(os.path.join(subdir_path, "*.ncu-rep"))
    
    if not report_files:
        print(f"No NCU report files found in {subdir_path}")
        return
    
    print(f"Found {len(report_files)} NCU report files in {subdir_path}")
    
    # Prepare CSV file path in the subdirectory
    output_file = os.path.join(subdir_path, "ncu_metrics.csv")
    
    # Import necessary modules from the provided path
    sys.path.append(ncu_report_module)
    
    try:
        import ncu_report
    except ImportError:
        print(f"Failed to import ncu_report module from {ncu_report_module}")
        return
    
    # Write metrics to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        header = [
            "Benchmark Type",
            "Test Name",
            "Kernel ID",
            "Thread block Limit SM",
            "Thread block limit registers",
            "Thread block limit shared memory",
            "Thread block limit warps",
            "Theoretical max active warps per SM",
            "Theoretical occupancy",
            "Achieved active warps per SM",
            "Achieved occupancy",
            "Unified L1 cache hit rate",
            "Unified L1 cache hit rate for read transactions (global memory accesses)",
            "L2 cache hit rate",
            "GMEM read requests",
            "GMEM write requests",
            "GMEM total requests",
            "GMEM read transactions",
            "GMEM write transactions",
            "GMEM total transactions",
            "Number of read transactions per read requests",
            "Number of write transactions per write requests",
            "L2 read transactions",
            "L2 write transactions",
            "L2 total transactions",
            "DRAM total transactions",
            "Total number of global atomic requests",
            "Total number of global reduction requests",
            "Global memory atomic and reduction transactions",
            "GPU active cycles",
            "SM active cycles",
            "Warp instructions executed",
            "Instructions executed per clock cycle (IPC)",
            "Total instructions executed per seconds (MIPS)",
            "Kernel execution time (ns)",
            "Unified L1 cache total requests",
            "Unified L2 cache total requests"
        ]
        csv_writer.writerow(header)
        
        # Process each report file
        for report_file_path in report_files:
            try:
                # Extract benchmark type (subdirectory name) and test name
                benchmark_type = os.path.basename(subdir_path) 
                test_name = os.path.splitext(os.path.basename(report_file_path))[0]
                
                print(f"Processing: {benchmark_type}/{test_name}")
                
                # Load the report
                report = ncu_report.load_report(report_file_path)
                kernel_num = len(report[0])
                
                for knum in range(kernel_num):
                    kernel = report[0][knum]
                    
                    # Initialize results with benchmark type and test name
                    results = [benchmark_type, test_name, str(knum)]
                    
                    # Collect metrics (same as original script)
                    results.append(get_launch__occupancy_limit_blocks(kernel))
                    results.append(get_launch__occupancy_limit_registers(kernel))
                    results.append(get_launch__occupancy_limit_shared_mem(kernel))
                    results.append(get_launch__occupancy_limit_warps(kernel))
                    results.append(get_sm__maximum_warps_avg_per_active_cycle(kernel))
                    results.append(get_sm__maximum_warps_per_active_cycle_pct(kernel))
                    results.append(get_sm__warps_active_avg_per_cycle_active(kernel))
                    results.append(get_sm__warps_active_avg_pct_of_peak_sustained_active(kernel))
                    results.append(get_l1tex__t_sector_hit_rate_pct(kernel))
                    results.append(get_l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate_pct(kernel))
                    results.append(get_lts__t_sector_hit_rate_pct(kernel))
                    
                    # Memory requests and transactions
                    gmem_read_req = get_l1tex__t_requests_pipe_lsu_mem_global_op_ld_sum(kernel)
                    gmem_write_req = get_l1tex__t_requests_pipe_lsu_mem_global_op_st_sum(kernel)
                    results.append(gmem_read_req)
                    results.append(gmem_write_req)
                    results.append(gmem_read_req + gmem_write_req)
                    
                    gmem_read_trans = get_l1tex__t_sectors_pipe_lsu_mem_global_op_ld_sum(kernel)
                    gmem_write_trans = get_l1tex__t_sectors_pipe_lsu_mem_global_op_st_sum(kernel)
                    results.append(gmem_read_trans)
                    results.append(gmem_write_trans)
                    results.append(gmem_read_trans + gmem_write_trans)
                    
                    # Calculate ratios
                    if gmem_read_req == 0:
                        results.append(0.)
                    else:
                        results.append(gmem_read_trans / gmem_read_req)
                        
                    if gmem_write_req == 0:
                        results.append(0.)
                    else:
                        results.append(gmem_write_trans / gmem_write_req)
                    
                    # L2 transactions
                    l2_read_trans = get_l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld_sum(kernel)
                    l2_write_trans = get_l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st_sum(kernel)
                    results.append(l2_read_trans)
                    results.append(l2_write_trans)
                    results.append(l2_read_trans + l2_write_trans)
                    
                    # DRAM transactions
                    results.append(get_dram__sectors_read_sum(kernel) + get_dram__sectors_write_sum(kernel))
                    
                    # Atomic and reduction operations
                    atom_req = get_l1tex__t_requests_pipe_lsu_mem_global_op_atom_sum(kernel)
                    red_req = get_l1tex__t_requests_pipe_lsu_mem_global_op_red_sum(kernel)
                    results.append(atom_req)
                    results.append(red_req)
                    results.append(atom_req + red_req)
                    
                    # Cycle and execution statistics
                    results.append(get_gpc__cycles_elapsed_max(kernel))
                    results.append(get_sm__cycles_active_avg(kernel))
                    results.append(get_smsp__inst_executed_sum(kernel))
                    
                    ipc = get_sm__inst_executed_avg_per_cycle_elapsed(kernel)
                    results.append(ipc)
                    results.append(ipc * get_gpc__cycles_elapsed_avg_per_second(kernel) * 1e-6)
                    results.append(get_gpu__time_duration_sum(kernel))
                    
                    # Additional metrics
                    results.append(get_L1_Total_Requests(kernel))
                    results.append(get_lts__t_requests_srcunit_tex_sum(kernel))
                    
                    # Write row to CSV
                    csv_writer.writerow(results)
                    
            except Exception as e:
                print(f"Error processing {report_file_path}: {e}")
    
    print(f"Metrics successfully exported to {output_file}")

# Main execution starts here
if __name__ == "__main__":
    args = parse_arguments()
    
    # Find the ncu_report.py module
    ncu_report_path = args.cuda_path if args.cuda_path else detect_cuda_path()
    if not ncu_report_path:
        sys.exit(1)
    
    # Process each subdirectory in the root directory
    root_dir = args.reports_dir
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in {root_dir}")
        sys.exit(1)
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    for subdir in subdirs:
        process_subdirectory(subdir, ncu_report_path)
    
    print(f"Processing complete. Metrics exported for {len(subdirs)} parameter combinations.")

