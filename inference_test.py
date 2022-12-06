from inference_dense import ml1_api
from inference_sparse import ml2_api

path = '/Users/seungmipark/code/RakshitTithi/ICYO/raw_data/us.png'

result_dense = ml1_api(path)
print(result_dense)

result_sparse = ml2_api(path)
print(result_sparse)

