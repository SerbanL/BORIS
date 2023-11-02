#pragma once

//enum for crystal classes to be used in MElastic module (determines elastic coefficients to use and equations)

enum CRYSTAL_ {

	CRYSTAL_CUBIC = 0,
	CRYSTAL_HEXAGONAL = 1,
	CRYSTAL_TETRAGONAL = 2,
	CRYSTAL_TRIGONAL = 3,
	CRYSTAL_TRIGONAL2 = 4,
	CRYSTAL_ORTHORHOMBIC = 5
};