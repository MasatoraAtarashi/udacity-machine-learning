import poi_id
f1s = []
recalls = []
precisions = []
for i in range(1,10):
    F1,recall,precision = poi_id.poi_id()
    f1s.append(F1)
    recalls.append(recall)
    precisions.append(precision)

print "precision: ", sum(precisions) / len(precisions)
print "recall: ", sum(recalls) / len(recalls)
print "F1score: ", sum(f1s) / len(f1s)
