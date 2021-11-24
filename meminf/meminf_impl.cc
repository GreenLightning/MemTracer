#include "meminf.h"
#include "meminf_data.h"
#include <string>
#include <iostream>

std::unordered_map<uint64_t, Meminf> meminfs;

void meminf_describe(void *ptr, int desc)
{
	meminfs[(uint64_t)ptr].desc = desc;
}
