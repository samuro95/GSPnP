def normalize_min_max(A):
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(A.size())
    return AA

