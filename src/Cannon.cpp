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
			M[idx(i, j, size)] = rand() % 1000;
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

int shiftLeft(double *&M, int size, const int blockSize, const int init) {
	double *aux = nullptr;
	aux = new double[size];
	int step = blockSize;

	for (int k = 0, int s = 0; k < size; k += blockSize, s++) {
		for (int i = k; i < (k + blockSize); i++) {
			if (init > 0)
				step = s * blockSize;
			for (int j = 0; j < size; j++)
				aux[j] = M[idx(i, ((j + step) % size), size)];
			for (int j = 0; j < size; j++)
				M[idx(i, j, size)] = aux[j];
		}
	}
	return 0;
}

int shiftUP(double *&M, int size, const int blockSize, const int init) {
	double *aux = nullptr;
	aux = new double[size];
	int step = blockSize;

	for (int k = 0, int s = 0; k < size; k += blockSize, s++) {
		for (int i = k; i < (k + blockSize); i++) {
			if (init > 0)
				step = s * blockSize;
			for (int j = 0; j < size; j++)
				aux[j] = M[idx(((j + step) % size), i, size)];
			for (int j = 0; j < size; j++)
				M[idx(j, i, size)] = aux[j];
		}
	}
	return 0;
}

int multProcessPar(double *&A, double *&B, double *&C, int size, int xblock) {
	int blockSize = size / xblock;
	int l, m, r, c, k, rbegin, rend, cbegin, cend, idThread;
	double *sa = nullptr;
	double *sb = nullptr;
	double *sc = nullptr;
#pragma omp parallel default(none) private(l, m, r, c, k, rbegin, rend, cbegin, cend, idThread, sa, sb, sc) shared(A, B, C) num_threads()
	{
		idThread = omp_get_thread_num();
		rbegin = (idThread / xblock) * blockSize;
		rend = rbegin + blockSize;

		cbegin = (idThread % xblock) * blockSize;
		cend = cbegin + blockSize;

		sa = new double[blockSize * blockSize];
		sb = new double[blockSize * blockSize];
		sc = new double[blockSize * blockSize];

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++) {
				sa[idx(l, m, blockSize)] = A[idx(r, c, size)];
				sb[idx(l, m, blockSize)] = B[idx(r, c, size)];
				sc[idx(l, m, blockSize)] = C[idx(r, c, size)];
			}
		}

		matrixMult(sa, sb, sc, blockSize);

		for (r = rbegin, l = 0; r < rend; r++, l++) {
			for (c = cbegin, m = 0; c < cend; c++, m++)
				C[idx(r, c, size)] = sc[idx(l, m, blockSize)];
		}
	}

	return 0;
}

int cannonPar(double *&A, double *&B, double *&C, int size, int xblock) {
	int blockSize = size / xblock;
	for (int i = 0; i < xblock; i++) {
		multProcessPar(A, B, C, size, xblock);
		shiftLeft(A, size, blockSize, 0);
		shiftUP(B, size, blockSize, 0);		
	}
	return 0;
}

int main(int argc, char** argv) {
	double *A = nullptr;
	double *B = nullptr;
	double *C = nullptr;
	int size = 0;
	
	size = atoi(argv[1]);

	srand((unsigned int)time(NULL));
	A = new double [size*size];
	B = new double [size*size];
	C = new double [size*size];
	matrixGenerate(A, size);
	matrixGenerate(B, size);
	
	//matrixPrint(A, size);
	//matrixPrint(B, size);

	double startTime = omp_get_wtime();
	matrixMult(A, B, C, size);
	double endTime = omp_get_wtime();
	
	//matrixPrint(C, size);
	cout << "Time: " << endTime - startTime << endl;

	delete []A;
	delete []B;
	delete []C;
	return 0;
}