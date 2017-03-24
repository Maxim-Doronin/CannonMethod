#include <omp.h>
#include <iostream>
#include <ctime> 
using namespace std;

inline int idx(int i, int j, int size) {
	return i*size + j;
}

int matrixGenerate(double *&M, const int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			M[idx(i, j, size)] = rand() % 100 - 50;
	return 0;
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

int matrixMult(double *&A, double *&B, double *&C, int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			C[idx(i, j, size)] = 0;
			for (int k = 0; k < size; k++)
				C[idx(i, j, size)] += A[idx(i, k, size)] * B[idx(k, j, size)];
		}
	return 0;
}

int multProcessPar(double *&A, double *&B, double *&C, const int size, const int numthreads, int i) {
	int blockSize = size / numthreads;
	int l, m, r, c, k, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
#pragma omp parallel default(none) private(l, m, r, c, k, rbegin, rend, cbegin, cend, idThread, sa, sb, sc) shared(A, B, C, size, blockSize, numthreads, i) num_threads(numthreads * numthreads)
	{
		idThread = omp_get_thread_num();
		rbegin = (idThread / numthreads) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % numthreads) * blockSize;
		cend = cbegin + blockSize;

		sa = new double[blockSize * blockSize];
		sb = new double[blockSize * blockSize];
		sc = new double[blockSize * blockSize];

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++) {
				sa[idx(l, m, blockSize)] = A[idx(r, (c + i*blockSize + (idThread / numthreads) * blockSize) % size, size)];
				sb[idx(l, m, blockSize)] = B[idx((r + i*blockSize + (idThread % numthreads) * blockSize) % size, c, size)];
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

	return 0;
}

int multProcessCon(double *&A, double *&B, double *&C, int size, int numthreads, int i) {
	int blockSize = size / numthreads;
	int l, m, r, c, k, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
	for (idThread = 0; idThread < numthreads * numthreads; idThread++) {
		rbegin = (idThread / numthreads) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % numthreads) * blockSize;
		cend = cbegin + blockSize;

		sa = new double[blockSize * blockSize];
		sb = new double[blockSize * blockSize];
		sc = new double[blockSize * blockSize];

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++) {
				sa[idx(l, m, blockSize)] = A[idx(r, (c + i*blockSize + (idThread / numthreads) * blockSize) % size, size)];
				sb[idx(l, m, blockSize)] = B[idx((r + i*blockSize + (idThread % numthreads) * blockSize) % size, c, size)];
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

int cannonPar(double *&A, double *&B, double *&C, int size, int numthreads) {
	int blockSize = size / numthreads;
	//shiftLeft(A, size, blockSize, 1);
	//shiftUp(B, size, blockSize, 1);
	for (int i = 0; i < numthreads; i++) {
		multProcessPar(A, B, C, size, numthreads, i);
		//shiftLeft(A, size, blockSize, 0);
		//shiftUp(B, size, blockSize, 0);	
	}
	//shiftRight(A, size, blockSize, numthreads);
	//shiftDown(B, size, blockSize, numthreads);
	return 0;
}

int cannonCon(double *&A, double *&B, double *&C, int size, int numthreads) {
	int blockSize = size / numthreads;
	//shiftLeft(A, size, blockSize, 1);
	//shiftUp(B, size, blockSize, 1);
	for (int i = 0; i < numthreads; i++) {
		multProcessCon(A, B, C, size, numthreads, i);
		//shiftLeft(A, size, blockSize, 0);
		//shiftUp(B, size, blockSize, 0);
	}
	//shiftRight(A, size, blockSize, numthreads);
	//shiftDown(B, size, blockSize, numthreads);
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
	int size = 0;
	int numthreads = 1;
	size = atoi(argv[1]);
	if (argc == 3)
		numthreads = atoi(argv[2]);
	if (size%numthreads != 0) {
		cout << "Error!" << endl;
		return -1;
	}

	cout << endl;
	cout << "---------Cannon algorithm for matrix multiplication-----------" << endl << endl;
	cout << "Matrix size: " << size << endl;
	cout << "Number of blocks: " << numthreads * numthreads << endl;
	
	srand((unsigned int)time(NULL));
	A = new double [size*size];
	B = new double [size*size];
	C = new double [size*size];
	C1 = new double[size*size];
	C2 = new double[size*size];
	for (int i = 0; i < size * size; i++) {
		C[i] = 0;
		C1[i] = 0;
		C2[i] = 0;
	}
	matrixGenerate(A, size);
	matrixGenerate(B, size);
	
	double startTimeTM = omp_get_wtime();
	matrixMult(A, B, C, size);
	double endTimeTM = omp_get_wtime();
	cout << "Trivial mult time: " << endTimeTM - startTimeTM << endl;

	double startTimeCC = omp_get_wtime();
	cannonCon(A, B, C2, size, numthreads);
	double endTimeCC = omp_get_wtime();
	cout << "Cannon consistent time: " << endTimeCC - startTimeCC << endl;

	double startTimeCP = omp_get_wtime();
	cannonPar(A, B, C1, size, numthreads);
	double endTimeCP = omp_get_wtime();
	cout << "Cannon parallel time: " << endTimeCP - startTimeCP << endl;

	cout << "Boost: " << (endTimeCC - startTimeCC) / (endTimeCP - startTimeCP) << endl;
	cout << "Error: " << matrixCond(C1, C2, size) << endl;

	delete []A;
	delete []B;
	delete []C;
	delete[]C1;
	delete[]C2;
	return 0;
}