import pickle

path4 = "data/boxnet/labels4/"
path5 = "data/boxnet/labels5/"
path6 = "data/boxnet/labels6/"

with open(path4 + 'centroids_list.pickle', 'rb') as handle:
    centroids_list4 = pickle.load(handle)
with open(path5 + 'centroids_list5.pickle', 'rb') as handle:
    centroids_list5 = pickle.load(handle)
with open(path6 + 'centroids_list.pickle', 'rb') as handle:
    centroids_list6 = pickle.load(handle)
with open(path4 + 'uri_list.pickle', 'rb') as handle:
    uri_list4 = pickle.load(handle)
with open(path5 + 'uri_list5.pickle', 'rb') as handle:
    uri_list5 = pickle.load(handle)
with open(path6 + 'uri_list.pickle', 'rb') as handle:
    uri_list6 = pickle.load(handle)
print("hello")