import pickle_man

picman = pickle_man.PickleMan("registered_feature/feature_50.pkl")
# data,_ = picman.get_data()
#print(data)

picman.combine("registered_feature/feature_60.pkl")
# print(picman.get_data())
print(picman.get_person_num())
persons , _= picman.get_data()
print(persons)

for person in ['Moran_Atias', "Abraham_Benrubi", 'Hector_Elizondo', 'Ben_Stiller', 'Sarah_Palin', 'Vincent_Pastore', 'Raoul_Bova', 'Desmond_Harrington', 'Mike_Epps']:
    print(picman.delete(person))

print(picman.get_person_num())
picman.save_data("registered_feature/feature_100.pkl")
