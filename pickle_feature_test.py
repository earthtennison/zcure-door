import os,pickle
extracted_feat = []
feat_list = [f for f in os.listdir("registered_feature")]
for feature in feat_list:    
    with open("registered_feature/"+feature,'rb') as f: arrayname1 = pickle.load(f)
    extracted_feat.append(arrayname1)


#print(extracted_feat)
print(extracted_feat[1].shape)