#include "meminf.h"
#include "meminf_storage.h"
#include <string>
#include <iostream>

std::unordered_map<uint64_t, std::string> meminfs;

void meminf_describe(void *ptr, const char *desc)
{
	std::cout << "Received description for pointer " << ptr << ": " << desc << std::endl;
	meminfs[(uint64_t)ptr] = desc;
}
