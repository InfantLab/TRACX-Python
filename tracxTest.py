# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:48:08 2016

@author: pss02ca
"""

import tracx

tracx1 = tracx.Tracx()


#print(tracx1.fahlmanOffset)
#print(tracx1.bias)
#print(tracx1.sentenceRepetitions)


#kirkham et al. 2002
#tracx1.set_training_data("babefefcdabefabababefcdefefcdefabefabefababefcdefabcdefcdcdcdefefcdcdabefabefcdefcdcdababefcdabcdefcdefababcdabefefcdabefabcdefcdababefcdefcdefabefefcdcdcdcdefefababababefcdcdefefc")
#tracx1.set_tracking_words(["ab", "af"]) 
#tracx1.testWords = ["ab", "ef"] 
#tracx1.testPartWords= ["bc", "fa"] 
#tracx1.testNonWords = ["af", "ca"]

#Saffran et al. 1996
tracx1.set_training_data("ghidefghijklghidefabcdefabcdefabcghidefghiabcghiabcjklghiabcjklabcghijklghidefghidefjkldefabcghidefabcjklghidefjklghiabcdefjklghidefabcdefabcdefabcghiabcjklabcghiabcjklabcjklabcjkldefghidefabcghijklabcjklghiabcghidefjkldefabcdefghiabcghidefabcghiabcjkldefabcdefabcdefabcjkldefabcjkldefabcjklabcghidefjklghiabcghidefabcdefghiabcjkldefjklabcjkldefjklghijklabcdefghiabcjklghijklabcdefghidefghiabcjklghiabcghiabcdefghidefjklabcdefabcjklabcjklghidefjkldefghidefjkldefabcdefghiabcjklghiabcghiabcghijkldefghijklghiabcjklabcghijkldefabcghijklghidef")
tracx1.set_tracking_words(["ab", "af"]) 
tracx1.testWords = ["abc","def","ghi","jkl"] 
tracx1.testPartWords= ["cde","fgh","ijk","lab"] 
tracx1.testNonWords = ["aei","dhl","gkc","jbf"]

tracx1.create_input_encodings()
print(tracx1.get_input_encoding("b"))
print(tracx1.get_input_encoding("f"))
print(tracx1.get_unique_items())  

print(tracx1.inputWidth)

print(tracx1.decimal_to_binary(101))

tracx1.initialize_weights()

#print(tracx1.layer[0])

ret = tracx1.run_full_simulation()

print(ret) 

ret = tracx1.test_strings(["abc","def,ghi"] )
print(ret)