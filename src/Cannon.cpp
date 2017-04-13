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

class CannonTask {
private:
    int rowStartIdx;
    int colStartIdx;
    int size;
    double *A;
    double *B;
    double *C;
    int xSize;
public:
    CannonTask(double *_A, double *_B, double *_C,
        int _rowStartIdx, int _colStartIdx, int _size, int _xSize) :
        A(_A), B(_B), C(_C), rowStartIdx(_rowStartIdx), colStartIdx(_colStartIdx),
        size(_size), xSize(_xSize) {};
    void operator()() const {
        int blockSize = size / xSize;
        for (int x = 0; x < xSize; x++) {
            for (int i = rowStartIdx * blockSize; i < (rowStartIdx + 1)*blockSize; i++)
                for (int j = colStartIdx * blockSize; j < (colStartIdx + 1)*blockSize; j++)
                    for (int k = x * blockSize; k < (x + 1)*blockSize; k++) {
                        C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
                    }
       }
        
    }
};

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
	int l, m, r, c, k, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
#pragma omp parallel default(none) private(l, m, r, c, k, rbegin, rend, cbegin, cend, idThread, sa, sb, sc) shared(A, B, C, size, blockSize, xSize, i) num_threads(xSize * xSize)
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
				C[idx(r, c , size)] += sc[idx(l, m, blockSize)];
		}
		
		delete[]sa;
		delete[]sb;
		delete[]sc;
	}
	/*
	{
		idThread = omp_get_thread_num();
		rbegin = (idThread / numthreads) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % numthreads) * blockSize;
		cend = cbegin + blockSize;

		//sa = new double[blockSize * blockSize];
		//sb = new double[blockSize * blockSize];
		//sc = new double[blockSize * blockSize];

		int col = cbegin + i*blockSize + (idThread / numthreads) * blockSize;
		int row = rbegin + i*blockSize + (idThread % numthreads) * blockSize;
		for (r = rbegin; r < rend; r++) {
			for (c = cbegin; c < cend; c++) {
				//double a = A[idx(r, (c + i*blockSize + (idThread / numthreads) * blockSize) % size, size)];
				//double b = B[idx((r + i*blockSize + (idThread % numthreads) * blockSize) % size, c, size)];
				for (int k = 0; k < blockSize; k++) {
					double a = A[idx(r, (col + k) % size, size)];
					double b = B[idx((row + k) % size, c, size)];
					C[idx(r, c, size)] += a * b;
				}


			}
		}
		
		matrixMult(sa, sb, sc, blockSize);
		for (r = rbegin, l = 0; r < rend; r++, l++) {
		for (c = cbegin, m = 0; c < cend; c++, m++)
		C[idx(r, c , size)] += sc[idx(l, m, blockSize)];
		}
		

		//delete[]sa;
		//delete[]sb;
		//delete[]sc;
	} */

	return 0;
}

int multProcessCon(double *&A, double *&B, double *&C, int size, int xSize, int i) {
	int blockSize = size / xSize;
	int l, m, r, c, k, rbegin, rend, cbegin, cend, idThread;
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

int cannonTBB(double *&A, double *&B, double *&C, int size, int xSize) {
    task_group taskGroup;
    for (int i = 0; i < xSize; i++)
        for (int j = 0; j < xSize; j++)
            taskGroup.run(CannonTask(A, B, C, i, j, size, xSize));
    taskGroup.wait();
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
	int numthreads = 1;
	size = atoi(argv[1]);
	if (argc == 3)
		numthreads = atoi(argv[2]);
	if (size % (int)sqrt((int)numthreads) != 0) {
		cout << "Error!" << endl;
		return -1;
	}
    int xSize = (int)sqrt((int)numthreads);

	cout << endl;
	cout << "---------Cannon algorithm for matrix multiplication-----------" << endl << endl;
	cout << "Matrix size: " << size << endl;
	cout << "Number of blocks: " << xSize * xSize << endl;
	
	srand((unsigned int)time(NULL));
	A = new double [size*size];
	B = new double [size*size];
	C = new double [size*size];
	C1 = new double[size*size];
	C2 = new double[size*size];
    C3 = new double[size*size];
	for (int i = 0; i < size * size; i++) {
		C[i] = 0;
		C1[i] = 0;
		C2[i] = 0;
        C3[i] = 0;
	}
	matrixGenerate(A, size);
	matrixGenerate(B, size);
	
	double startTimeTM = omp_get_wtime();
	matrixMult(A, B, C, size);
	double endTimeTM = omp_get_wtime();
	cout << "Trivial mult time: " << endTimeTM - startTimeTM << endl;

	double startTimeCC = omp_get_wtime();
	cannonCon(A, B, C2, size, xSize);
	double endTimeCC = omp_get_wtime();
	cout << "Cannon consistent time: " << endTimeCC - startTimeCC << endl;
    
	double startTimeCP = omp_get_wtime();
	cannonPar(A, B, C1, size, xSize);
	double endTimeCP = omp_get_wtime();
	cout << "OpenMP parallel time: " << endTimeCP - startTimeCP << endl;
    
    task_scheduler_init init();
    double startTimeTBB = omp_get_wtime();
    cannonTBB(A, B, C3, size, xSize);
    double endTimeTBB = omp_get_wtime();
    cout << "TBB parallel time   : " << endTimeTBB - startTimeTBB << endl;

	cout << "Boost OpenMP: " << (endTimeCC - startTimeCC) / (endTimeCP - startTimeCP) << endl;
    cout << "Boost TBB   : " << (endTimeCC - startTimeCC) / (endTimeTBB - startTimeTBB) << endl;
	cout << "Error OpenMP: " << matrixCond(C1, C2, size) << endl;
    cout << "Error TBB   : " << matrixCond(C3, C2, size) << endl;

	delete []A;
	delete []B;
	delete []C;
	delete[]C1;
	delete[]C2;
	return 0;
}