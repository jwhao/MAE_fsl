import numpy as np

from sklearn.cluster import KMeans

def LRs_kmeans(support_f):   # query_f=none,sim=0.5
    """
    args:
    support_f:[h*w,c]
    query_f  :[h*w,c]
    output:
    support_kf:[k,c]
    query_kf  :[k,c]
    """
    # ss = support_f @ support_f.t()
    # ss = support_f @ support_f.T
    
    # sfset = []
    # sflist = []
    # for i,v in enumerate(ss):
    #     if i in sflist:
    #         continue
    #     sflist = []
    #     for j,vv in enumerate(v):
    #         if vv > sim:
    #             sflist.append(j)
        
    #     sfset = sflist if i==0 else sfset.extend(sflist)

    # n_clusters = len(sfset)

    s_kmeans = KMeans(n_clusters=2, random_state=10)
    s_kmeans.fit(support_f)
    s_labels = s_kmeans.labels_
    s_x1 = support_f[np.where(s_labels==0)]
    s_x2 = support_f[np.where(s_labels==1)]

    # q_kmeans = KMeans(n_clusters=2, random_state=10)
    # q_kmeans.fit(query_f)
    # q_labels = s_kmeans.labels_
    # q_x1 = query_f[np.where(s_labels==0)]
    # q_x2 = query_f[np.where(s_labels==1)]
    
    # sq = np.concatenate((support_f,query_f))
    # sq_kmeans = KMeans(n_clusters=2, random_state=10)
    # sq_kmeans.fit(sq)
    # sq_labels = sq_kmeans.labels_
    # sq_x1 = sq[np.where(sq_labels==0)]
    # sq_x2 = sq[np.where(sq_labels==1)]

    # length = np.min((len(s_x1),len(q_x1)))
    # support_kf = s_x1[:length]
    # query_kf = q_x1[:length]

    # return support_kf,query_kf
    return s_x1,s_x2

if __name__ == '__main__':
    sf = np.random.randn(196,64)
    qf = np.random.randn(196,64)

    # skf,qkf = LRs_kmeans(sf,qf)
    skf,qkf = LRs_kmeans(sf)