###############################################
#			Ismael Bonneau
###############################################

import elasticsearch as es
import ast
import pickle

req_path = "data/robust2004.txt"
indexname = "robust-2004"

engine = es.Elasticsearch([{'host':'localhost', 'port':9200}])
assert engine.ping()

def search(query, limit=2000):
	return engine.search(index=indexname, body={"_source": False,
	'size': limit,
	'query': {'match': {"text": query}}}, request_timeout=30)['hits']['hits']


with open(req_path, "r") as f:
	d_query = ast.literal_eval(f.read())

for k in d_query :
	d_query[k] = d_query[k][0] # On suppr les query langage naturel, et on garde que la query mot clé


results_bm25 = {}

for id_requete, query in d_query.items():
	results = search(query)
	results_bm25[id_requete] = [doc["_id"] for doc in results]
	print("query {}.. done.".format(id_requete))

pickle.dump(results_bm25, open("data/results_bm25_robust.pkl", 'wb'))


                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                  #                                                                       .(#%&&&&&&&&&&&%#(,                                                                                          
                  #                                                              (&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                                                                                  
                  #                                                         (&&&&&&&&&,                    .*%&&&&&&&&&&.                                                                            
                  #                                                    .%&&&&&&             .,,,,,,,,,             *%&&&&&&&,                                                                        
                  #                                                 (&&&&&       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,         %&&&&&&                                                                     
                  #                                              (&&&&     .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  ,,,,,,,,,,      .&&&&&&                                                                  
                  #                                           *&&&&&(    .,,,,,,,,,,,,,,,,,,,.,,,,,,.             .,,,,,,,,,      &&&&&&                                                               
                  #                                         #&&&&&     ,,,,,,,,,,,,,,,,,,,,,,,                     .,,,,,,,,,,,.     &&&&&,                                                            
                  #                                       #&&&&,    ,,,,,,,,,,,,,,,,,,,,,   .,                        ,,,,,,,,,,,,.    &&&&&,                                                          
                  #                                     &&&&&    .,,,,,,,,,,,,,,,,,,,. ,,,,                    ,,,.     ,,,,,,,,,,,,,    .&&&                                                        
                  #                                   #&&&&    ,,,,,,,,,,,,,,,,,,,,,,,                          .,,        ,,,,,,,,,,,,     &&&                                                      
                  #                                 ,&&&&    ,,,,,,,,,,,,,,,,,,,,,, ,,,                        ,,,,   ,,,     .,,,,,,,,,,,    &&&&,                                                    
                  #                                &&&&(   ,,,,,,,,,,,,,,,,,,,,,,,,,                          .,,      ,,,,       .,,,,,,,,,   *&&&&                                                   
                  #                               &&&&    ,,,,,,,,,,,,,,,,,,,,,,,,, .                                 ,,,,,,,         ,,,,,,,.   &&&&                                                  
                  #                             *&&&*   ,,,,,,,,,,,,,,,,,,,,,,,,,.                                     .,,,,             ,,,,,,   *&&&*                                                
                  #                            (&&&,   ,,,,,,,,,,,,,,,,,,,,,,,,, ,                                           ,,,,          ,,,,,,  .&&                                               
                  #                           %&&&.   ,,,,,,,,,,,,,,,,,,,,,,,,,,                             ,,,,,,,,,                      .,,,,.   &&&%                                              
                  #                          ,&&&   ,,,,,,,,,,,,,,,,,,,,,.,,.,                                 ,,,,,,,,..,.                  .,,,,.   &&&%                                             
                  #                          &&&,  .,,,,,,,,,,,,,,,,,  ,,  .                                    ,,,,,,,,, .  .,,,  ,         ,,,,,,.  *&&&(                                            
                  #                         &&&(   ,,,,,,,,,   ,,                            ,,..             ..,,,,,,,,, , , , ,      ,,. ,,,,,,,,,   (&&&                                            
                  #                        (&&&   ,,,,,    ,                                 ,,,,.             ,.,,,,,,,,,,,,,,,, . ,.,  ,,,,,,,,,,,,   &&&%                                           
                  #                        &&&      ,,,                           ..  ,  .,,,,,,,,,,            .,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  .&&&                                           
                  #                       %&&&                                    ,, ., ,.,,,,,,,,,,,,,.          , , ,,,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,   %&&%                                          
                  #                       &&&*                             .     ,,, , ,,,,,  ,,,,,,,,,,,,.        , . ,. .,,,,,,,,,,,,,,,,,,,,,,,,,,,,  ,&&&                                          
                  #                      (&&&                               ,,   ,,,,.,,,,,,    ,,,,,,,,,,,,,,       ., .  . ,,,,,,,,,,,,,,,,,,,,,,,,,,   &&&*                                         
                  #                      &&&&                                .  ,,,,,,,,,,,,  .,,,,,,,,,,,,,,,,,        ,,   , ,,,,,,,,,,,,,,,,,,,,,,,,   &&&&                                         
                  #                      &&                            .,     ,,,.,,,,,,.   ,,    ,,,,,,,,,,,,,,,          ,,,,,,,,,,,,,,,,,,,,,,,,,,   #&&&                                         
                  #                      &&                               ,    .           ,,   ,,.,,,,,,,,,,,,,,,,,.   .,,,,,,,,,,,,,,,,,,,,,,,,,,,,   #&&&                                         
                  #                      &&                       .        ,,       ,,,,,,       .,. ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,   #&&&,,,,,,,,,,,,,,,,,,,,.                    
                  # &&&&&&&&&&&&&&&&&&&&&&&                  ,,  ,,, .,,,,    .             ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,   #&&&&&&&&&&&&&&&&&&&&&&                    
                  # &&&&&&&&&&&&&&&&&&&&&&&               ,,  ,. ,,,. ,,,, ,            ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,   .,,,,,,,,,,,,,,,,,,,,&&                    
                  # &&&&                                                                                                                                                       &&                    
                  # &&&&       %#,                                                                                   ,@@@@@@@@&                                                &&                    
                  # &&&&       @@@(@@@                                ,,,,,,.      ##       &.   ,,,,,       @        @@%@@@@/(@@*     @@@@@@@   .@@@@@@@#    ,@@@@#  #@#      &&                    
                  # &&&&       @@@,@@@ @@&&@,   @@/@@,@@,,*/@@@@     %@(///@@    #@@%@@*  @@@@@@ @@(@@    #@@&@&      @@%@@%*@@%(@@   ,@/,,,@@,  ,@%%@@@*%@@  ,@%@@% .@@*@@    &&                    
                  # &&&&       @@@,@@% &@%%@,   @@/@@ @@/@@@@@*@@#   @@%@@@%@,  &@@*@@# *@@/@@@* %@/@%  .@@%#@@       @@#@@%  #@%#@@  @@/@@@/@%  ,@%%@,@@@/@@*.@%@@% @@/@@     &&                    
                  # &&&&       @@@,@@% %@%%@,   @@/@@ @@/@%  @@#@@# ,@%@@&@/@% %@@,@@   @@%@@%   %@/@% ,@@/@@         @@/@@%   @@,@@, @#@@.@/@@  ,@%%@   @@,@@,@&&@&@@/@@      &&                    
                  # &&&&       @@@,@@% %@%%@,   @@/@@ @@/@*  %@@/@% @@/@, @#@@ ,@@*@@   @@&&@*   %@/@% @@/@@          @@/@@%   @@,@@,%@/@* %@%@, ,@%%@   @@,@@ @@%@@@(@#       &&                    
                  # &&&&       @@@,@@% ,@%%@,   @@/@@ @@%@,  @@(@@  @%&@  %@&@. *@@,@@   @@/&@#  %@/@% @&@@,          @@/@@%   @@*@@ @@&@  ,@,@% ,@%%@ (@@&@@* @@%@@,@@        &&                    
                  # &&&&       %@@,@@, ,@%%@    @@/@@ @@%@@@@@*@@  &@*@,  ,@/@%   @@(&@#  #@%(@@ %@/@%,@%@@,          @@/@@% /@@/@@#*@/@%   @%%@ ,@%%@@@&/@*   @@%/@@%@@       &&                    
                  # &&&&       %@@,@@, ,@%%@    @@,@% @@%@@#,#@%   @%%@@@@@@/@@    @@,@@,  @@/@@%%@/@% @@(@&          @@/@@@@@%/@@* @@/@@@@@@@,@#,@%,/&@@/@#   @@/@@#&@.     &&                    
                  # &&&&       %@@,@@, ,@%%@    @@,@% @@%@@@@#%@  %@,@@@@@@@@(@*  ,@#(@@   @@/@@ %@/@% ,@&(@@         @@/#/,#@@@#  ,@/@@@@@@@@%%@,@%%@#&@@(@.  @@/@% #@&(@*    &&                    
                  # &&&&       %@@,@@, ,@%#@@@@@@@,@% @@%@% @@(&@ @%%@%    %@*@@ @@(#@@  @@%#@@  %@/@%  *@@/@@@       @@/%@@#      @@,@%    ,@@/@@@%%@, &@%@@  @@/@%  #@@/@@   &&                    
                  # &&&&  (%%%&@@@/@@,  #@@@(,/(@@/@% @@%@%  @@/@@@,@@     ,@%%@@@&@@*  @@@@@*   %@/@%    @@@@@@      @@/%@%      ,@(%@,     @@/&@@@@@,  *@&@@ @@@@%   .@@@    &&                    
                  # &&&&  %@@@@&//@@@     .&@@@.@@@@% @@@@%   @@#@@@@@      @@@@& #@      *@     (%%%(      .@       ,@@/%@%                                                   &&                    
                  # &&&&  ,@@@@@@@@@                                                                                 ,@@@@@%                                                   &&                    
                  # &&&&    ,,,,,                                                                                                                                              &&                    
                  # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&    ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                    
                  # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&,    ,,,,,,,,,,,,,     .,....,, .   ,,,,,,.    ,,,.  ,,,,,,,,,,,,,.    ,&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                    
                  #                                        &&&&&&&     ,,,,  .,,,,           ..         .            .,,,,,,,,.     &&&&&&&                                                            
                  #                                           &&&&&&&               .,..      .                 .               .&&&&&&&                                                               
                  #                                              &&&&&&&%                                                    %&&&&&&&                                                                  
                  #                                                 &&&&&&&&(                                            %&&&&&&&&                                                                     
                  #                                                    &&&&&&&&&&&(.                              ,%&&&&&&&&&&                                                                         
                  #                                                         &&&&&&&&&&&&&*,           .,#&&&&&&&&&&&&&&&                                                                             
                  #                                                              &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&                                                                                  
                  #                                                                       &&&&&&&&&&&&&&&&&                                                                                            
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     