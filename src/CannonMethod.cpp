#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <ctime>
#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"

using namespace std;
using namespace tbb;

inline int idx(int i, int j, int size) {
	return i*size + j;
}

int matrixPrint(double *&M, const int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			cout << M[idx(i, j, size)] << "\t";
		cout << endl;
	}
	cout << endl;
	return 0;
}

int matrixGenerate(double *&M, const int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			M[idx(i, j, size)] = rand() % 100 - 50;
	return 0;
}

int matrixMult(double *&A, double *&B, double *&C, int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			C[idx(i, j, size)] = 0;
			for (int k = 0; k < size; k++)
				C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
		}
	return 0;
}

int multProcessPar(double *&A, double *&B, double *&C, const int size, const int xSize, int i) {
	int blockSize = size / xSize;
	int l, m, r, c, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
#pragma omp parallel default(none) private(l, m, r, c, rbegin, rend, cbegin, cend, idThread, sa, sb, sc) shared(A, B, C, size, blockSize, xSize, i) num_threads(xSize * xSize)
	{
		idThread = omp_get_thread_num();
		rbegin = (idThread / xSize) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % xSize) * blockSize;
		cend = cbegin + blockSize;

		sa = new double[blockSize * blockSize];
		sb = new double[blockSize * blockSize];
		sc = new double[blockSize * blockSize];

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++) {
				sa[idx(l, m, blockSize)] = A[idx(r, (c + i*blockSize + (idThread / xSize) * blockSize) % size, size)];
				sb[idx(l, m, blockSize)] = B[idx((r + i*blockSize + (idThread % xSize) * blockSize) % size, c, size)];
			}
		}

		matrixMult(sa, sb, sc, blockSize);
		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++)
				C[idx(r, c, size)] += sc[idx(l, m, blockSize)];
		}

		delete[]sa;
		delete[]sb;
		delete[]sc;
	}
	return 0;
}

int multProcessCon(double *&A, double *&B, double *&C, int size, int xSize, int i) {
	int blockSize = size / xSize;
	int l, m, r, c, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
	for (idThread = 0; idThread < xSize * xSize; idThread++) {
		rbegin = (idThread / xSize) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % xSize) * blockSize;
		cend = cbegin + blockSize;

		sa = new double[blockSize * blockSize];
		sb = new double[blockSize * blockSize];
		sc = new double[blockSize * blockSize];

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++) {
				sa[idx(l, m, blockSize)] = A[idx(r, (c + i*blockSize + (idThread / xSize) * blockSize) % size, size)];
				sb[idx(l, m, blockSize)] = B[idx((r + i*blockSize + (idThread % xSize) * blockSize) % size, c, size)];
			}
		}

		matrixMult(sa, sb, sc, blockSize);

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++)
				C[idx(r, c, size)] += sc[idx(l, m, blockSize)];
		}

		delete[]sa;
		delete[]sb;
		delete[]sc;
	}
	return 0;
}

class CannonTask {
private:
	int rowStartIdx;
	int colStartIdx;
	int size;
	double *A;
	double *B;
	double *C;
	int blockSize;
	int xSize;
public:
	CannonTask(double *_A, double *_B, double *_C,
		int _rowStartIdx, int _colStartIdx, int _size, int _blockSize, int _xSize) :
		A(_A), B(_B), C(_C), rowStartIdx(_rowStartIdx), colStartIdx(_colStartIdx),
		size(_size), blockSize(_blockSize), xSize(_xSize) {};
	void operator()() const {
		for (int x = 0; x < xSize; x++)
			for (int i = rowStartIdx * blockSize; i < (rowStartIdx + 1)*blockSize; i++)
				for (int j = colStartIdx * blockSize; j < (colStartIdx + 1)*blockSize; j++)
					for (int k = x * blockSize; k < (x + 1)*blockSize; k++) {
						C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
					}
	}
};

int cannonTBB(double *&A, double *&B, double *&C, int size, int xSize, task_group &taskGroup) {
	int blockSize = size / xSize;
	for (int i = 0; i < xSize; i++)
		for (int j = 0; j < xSize; j++)
			taskGroup.run(CannonTask(A, B, C, i, j, size, blockSize, xSize));
	taskGroup.wait();
	return 0;
}


int cannonPar(double *&A, double *&B, double *&C, int size, int xSize) {
	int blockSize = size / xSize;
	for (int i = 0; i < xSize; i++) {
		multProcessPar(A, B, C, size, xSize, i);
	}
	return 0;
}

int cannonCon(double *&A, double *&B, double *&C, int size, int xSize) {
	int blockSize = size / xSize;
	for (int i = 0; i < xSize; i++) {
		multProcessCon(A, B, C, size, xSize, i);
	}
	return 0;
}

double matrixCond(double *&A, double *&B, int size) {
	double error = 0;
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			error += abs(A[idx(i, j, size)] - B[idx(i, j, size)]);
	return error;
}

int main(int argc, char** argv) {
	double *A = nullptr;
	double *B = nullptr;
	double *C = nullptr;
	double *C1 = nullptr;
	double *C2 = nullptr;
	double *C3 = nullptr;
	int size = 0;
	int root = 0;
	int myRank;
	int rankSize;
	int numthreads = 1;
	int xSizeProc = 1;
	int xSizeThread = 1;
	int flag = 1;

	double startTimeMPI = 0;
	double endTimeMPI = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm cart2d;
	MPI_Status status;
	MPI_Datatype block_t;

	MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	if (myRank == root) {
		size = atoi(argv[1]);
		if (argc == 3) {
			numthreads = atoi(argv[2]);
		}

		if (size % (int)sqrt(rankSize) != 0) {
			cout << "Error! Size must be divided into sqrt(number of processes)" << endl;
			flag = 0;
		}
		if ((size / (int)sqrt(rankSize)) % (int)sqrt(numthreads) != 0) {
			cout << "Error! Size of block must be divided into sqrt(number of threads)" << endl;
			flag = 0;
		}
		xSizeProc = (int)sqrt((int)rankSize);
		xSizeThread = (int)sqrt((int)numthreads);

		if (xSizeProc * xSizeProc != rankSize ||
			xSizeThread * xSizeThread != numthreads) {
			cout << "Error! Number of processes and number of threads must be completed square" << endl;
			flag = 0;
		}
	}
	MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
	if (flag == 0) {
		MPI_Finalize();
		return 0;
	}
	MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&numthreads, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&xSizeProc, 1, MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&xSizeThread, 1, MPI_INT, root, MPI_COMM_WORLD);

	

	int dims[2] = { xSizeProc , xSizeProc };
    int periods[2] = { 1, 1 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart2d);
    MPI_Comm_rank(cart2d, &myRank);
    int root_coords[2] = { 0, 0 };
    MPI_Cart_rank(cart2d, root_coords, &root);

    int leftRank, rightRank, upRank, downRank;
    MPI_Cart_shift(cart2d, 1, -1, &rightRank, &leftRank);
    MPI_Cart_shift(cart2d, 0, -1, &downRank, &upRank);

	//Allocate memory for each process
    int blockSize = size / xSizeProc;
    double *Ablock = new double [blockSize * blockSize];
	double *Bblock = new double [blockSize * blockSize];
	double *Cblock = new double [blockSize * blockSize];

    MPI_Datatype temptype;
    MPI_Type_vector(blockSize, blockSize, size, MPI_DOUBLE, &temptype);
    MPI_Type_create_resized(temptype, 0, sizeof(double), &block_t);
    MPI_Type_commit(&block_t);

    if (myRank == root) {
		cout << endl;
		cout << "---------Cannon algorithm for matrix multiplication-----------" << endl << endl;
		cout << "Matrix size: " << size << " x " << size << endl;
		cout << "Number of process blocks: " << rankSize << endl;
		cout << "Number of thread blocks : " << numthreads << endl;

        srand((unsigned int)time(NULL));
    	A = new double[size*size];
    	B = new double[size*size];
    	C = new double[size*size];  // MPI + OpenMP
    	C1 = new double[size*size];  // Последовательно
		C2 = new double[size*size];  // Только OpenMP
		C3 = new double[size*size];  // Только TBB
    	for (int i = 0; i < size * size; i++) {
    		C[i] = 0;
    		C1[i] = 0;
			C2[i] = 0;
			C3[i] = 0;
    	}
    	matrixGenerate(A, size);
    	matrixGenerate(B, size);
        startTimeMPI = MPI_Wtime();

        for (int i = 0; i < blockSize; ++i)
            for (int j = 0; j < blockSize; j++) {
                Ablock[idx(i, j, blockSize)] = A[idx(i, j, size)];
                Bblock[idx(i, j, blockSize)] = B[idx(i, j, size)];
            }

		// Рассылка матрицы А
        for (int i = 0; i < xSizeProc; i++)
            for (int j = 0; j < xSizeProc; j++)
                if ((i != 0) || (j != 0)) {
                    int dest;
                    int block_coords[2] = { i, j - i };
                    if (block_coords[1] < 0)
                        block_coords[1] += xSizeProc;
                    MPI_Cart_rank(cart2d, block_coords, &dest);
                    MPI_Send(&A[i * size * blockSize + j * blockSize], 1, block_t, dest, 0, cart2d);
                }

		// Рассылка матрицы B
        for (int i = 0; i < xSizeProc; i++)
            for (int j = 0; j < xSizeProc; j++)
                if ((i != 0) || (j != 0)) {
                    int dest;
                    int block_coords[2] = { i - j, j };
                    if (block_coords[0] < 0)
                        block_coords[0] += xSizeProc;
                    MPI_Cart_rank(cart2d, block_coords, &dest);
                    MPI_Send(&B[i * size * blockSize + j * blockSize], 1, block_t, dest, 1, cart2d);
                }
    } else {
		// Прием блоков от root
        MPI_Recv(Ablock, blockSize * blockSize, MPI_DOUBLE, root, 0, cart2d, &status);
        MPI_Recv(Bblock, blockSize * blockSize, MPI_DOUBLE, root, 1, cart2d, &status);
    }

	// Начало вычислений 
    cannonPar(Ablock, Bblock, Cblock, blockSize, xSizeThread);
    for (int i = 1; i < xSizeProc; i++) {
		// Циклические сдвиги
        MPI_Sendrecv_replace(Ablock, blockSize * blockSize, MPI_DOUBLE, leftRank, 0, rightRank, 0, cart2d, &status);
        MPI_Sendrecv_replace(Bblock, blockSize * blockSize, MPI_DOUBLE, upRank, 1, downRank, 1, cart2d, &status);
        cannonPar(Ablock, Bblock, Cblock, blockSize, xSizeThread);
    }


    if (myRank == root) {
        for (int i = 0; i < blockSize; i++)
            for (int j = 0; j < blockSize; j++)
                C[idx(i, j, size)] = Cblock[idx(i, j, blockSize)];

		// Ожидание результатов от всех процессов
        for (int i = 0; i < xSizeProc; i++)
            for (int j = 0; j < xSizeProc; j++)
                if ((i != 0) || (j != 0)) {
                    int source;
                    int block_coords[2] = { i, j };
                    MPI_Cart_rank(cart2d, block_coords, &source);
                    MPI_Recv(&C[i * size * blockSize + j * blockSize], 1, block_t, source, 4, cart2d, &status);
                }

        endTimeMPI = MPI_Wtime();

        cout << "MPI + OpenMP time     : " << endTimeMPI - startTimeMPI << endl;

		// Ппоследовательные вычисления
        double startTimeCC = omp_get_wtime();
    	cannonCon(A, B, C1, size, xSizeProc);
    	double endTimeCC = omp_get_wtime();
    	cout << "Cannon consistent time: " << endTimeCC - startTimeCC << endl;

		// Только OpenMP
		double startTimeOpenMP = omp_get_wtime();
		cannonPar(A, B, C2, size, xSizeProc);
		double endTimeOpenMP = omp_get_wtime();
		cout << "OpenMP time           : " << endTimeOpenMP - startTimeOpenMP << endl;

		// Только TBB
		task_scheduler_init init(numthreads);
		task_group taskGroup;
		double startTimeTBB = omp_get_wtime();
		cannonTBB(A, B, C3, size, xSizeProc, taskGroup);
		double endTimeTBB = omp_get_wtime();
		cout << "TBB time              : " << endTimeTBB - startTimeTBB << endl;

		// Сравнение результатов
        cout << "Boost MPI + OpenMP    : " << (endTimeCC - startTimeCC) / (endTimeMPI - startTimeMPI) << endl;
		cout << "Boost only OpenMP     : " << (endTimeCC - startTimeCC) / (endTimeOpenMP - startTimeOpenMP) << endl;
		cout << "Boost only TBB        : " << (endTimeCC - startTimeCC) / (endTimeTBB - startTimeTBB) << endl;
    	cout << "Error                 : " << matrixCond(C, C1, size) << endl;
        delete[]A;
    	delete[]B;
    	delete[]C;
    	delete[]C1;
    } else {
		// Отправка результатов на root
        MPI_Send(Cblock, blockSize * blockSize, MPI_DOUBLE, root, 4, cart2d);
    }

    delete[] Ablock;
    delete[] Bblock;
    delete[] Cblock;

    MPI_Type_free(&block_t);
    MPI_Comm_free(&cart2d);
	MPI_Finalize();

	return 0;
}

