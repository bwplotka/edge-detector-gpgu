#include "MPIEdgeDetector.hpp"

int main(int argc, char **argv) {
	int my_id;
	int all_processes;
	int ret;
	// Initialize MPI
	MPI_Init(&argc, &argv);
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &all_processes);
	if (!my_id) LOG1("Started MPI workload - processes: %d\n", all_processes)

	ret = run_process(my_id, all_processes);
	MPI_Finalize();
	return ret;
}