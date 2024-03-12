import pymongo
client=pymongo.MongoClient('mongodb+srv://bikubikusingh1211:12345678S@cluster0.tffvkz6.mongodb.net/')
mydb=client['studentinfo']
collection=mydb.studentinfo


# information.insert_one(record)
# record=[{
#     'firstname':'Viku',
#     'lastname':'Singh1',
#     'age':'25'
# },{
#     'firstname':'Viku',
#     'lastname':'Singh2',
#     'age':'24'
# },{
#     'firstname':'Viku',
#     'lastname':'Singh3',
#     'age':'26',
#     'department':'EE'
# },{
#     'firstname':'Viku',
#     'lastname':'Singh1',
#     'age':'25','sex':'male'
# }]
# information.insert_many(record)
# Perform the query and print data only once
# cursor = information.find()

# for document in cursor:
#     print(document)

# for i in information.find({'lastname':'Singh2'}):
#     print(i)

# Query documents using query operators($in,$lt,$gt)
# for record in information.find({'lastname':{'$in':['Singh1','Singh2']}}):
#     print(record)

data=[
    {"user":"Krish","Subject":"Database","score":80},
    {"user":"Amit","Subject":"javascript","score":90},
    {"user":"Amit","Subject":"Database","score":85},
    {"user":"Krish","Subject":"javascript","score":75},
    {"user":"Amit","Subject":"Database","score":60},
    {"user":"Krish","Subject":"Database","score":95},
    {"user":"Krish88","Subject":"C++","score":90}
]  
collection.insert_many(data)


# Total no.of records
# agg_result=collection.aggregate(
#     [{
#         "$group" :
#         {"_id" : "$Subject",
#          "Total Subject" : {"$sum":1}}
         
#     }]
# )
# for i in agg_result:
#     print(i)
#     import datetime as datetime