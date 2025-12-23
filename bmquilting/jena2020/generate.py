import numpy as np


inf = float('inf')


def getMinCutPatchHorizontal(ref_block, patch_block, blocksize, overlap):
	'''
	Get the min cut patch done horizontally
	'''
	err = ((ref_block[:, :overlap] - patch_block[:, :overlap]) ** 2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask = np.zeros((blocksize, blocksize, ref_block.shape[2]), dtype=np.float32)
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1

	res_block = np.zeros(ref_block.shape)
	res_block[:, :overlap] = ref_block[:, :overlap]
	res_block = res_block * mask + patch_block * (1 - mask)
	return res_block, mask[:, :, 0]


def getMinCutPatchVertical(ref_block, patch_block, blocksize, overlap):
	'''
	Get the min cut patch done vertically
	'''
	res_block, mask = getMinCutPatchHorizontal(np.rot90(ref_block), np.rot90(patch_block), blocksize, overlap)
	return np.rot90(res_block, 3), mask


def getMinCutPatchBoth(ref_block, patch_block, blocksize, overlap):
	'''
	Find minCut for both and calculate
	'''
	err = ((ref_block[:, :overlap] - patch_block[:, :overlap]) ** 2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask1 = np.zeros((blocksize, blocksize, patch_block.shape[2]), dtype=np.float32)
	for i in range(len(path)):
		mask1[i, :path[i]+1] = 1

	###################################################################
	## Now for vertical one
	err = ((np.rot90(ref_block)[:, :overlap] - np.rot90(patch_block)[:, :overlap]) ** 2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask2 = np.zeros((blocksize, blocksize, patch_block.shape[2]), dtype=np.float32)
	for i in range(len(path)):
		mask2[i, :path[i]+1] = 1
	mask2 = np.rot90(mask2, 3)

	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

	# Put first mask
	res_block = np.zeros(patch_block.shape)
	res_block[:, :overlap] = mask1[:, :overlap]*ref_block[:, :overlap]
	res_block[:overlap, :] = res_block[:overlap, :] + mask2[:overlap, :]*ref_block[:overlap, :]
	res_block = res_block + (1-np.maximum(mask1, mask2)) * patch_block
	return res_block, mask2[:, :, 0]
