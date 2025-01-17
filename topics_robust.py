###############################################
#			Ismael Bonneau
###############################################


import os

#fichier csv contenant les queries et une suite de mots clés + concepts. Les concepts sont identifiés par un $#! au début
topics_queries_file = "data/topics-title.annotated.csv"

augmented_queries = {} #dict str -> dict str -> list(str)

#contient les id des queries en clé, avec en valeur un dictionnaire qui contient 2 clés: "keywords" et "concepts" avec en valeur 2 listes
#par exemple:
# {'303': {'keywords': ['hubble', 'telescope', 'achievements'], 'concepts': ['telescope.n.01', 'accomplishment.n.01']}}


with open(topics_queries_file, "r") as f:
	for line in f:
		augmented_queries[line.strip().split("\t")[0]] = {}
		concepts = [w.replace("$#!", '') for w in line.strip().split("\t")[1].split(" ") if "$#!" in w]
		keywords = [w for w in line.strip().split("\t")[1].split(" ") if "$#!" not  in w]
		
		augmented_queries[line.strip().split("\t")[0]]["concepts"] = concepts
		augmented_queries[line.strip().split("\t")[0]]["keywords"] = keywords


                                                                                                                                                                    
                                                                                                                                                                                                     
                                                                                                                                                                                                     
                                                                                                                                                                                                     
#                                                                               .,***********                                                                                                          
#                                                                              **************                                                                                                          
#                                                                               *************,                                                                                                         
#                                                                               .************,                                                                                                         
#                                                                                .************                                                                                                         
#                                                                                 ************                                                                                                         
#                                                                                  ***********,                                                                                                        
#                                                                                  .***********                                                                                                        
#                                                                                   ***********.                                                                                                       
#                                                                                    ***********                                                                                                       
#                                                                                    ,**********.                                                                                                      
#                                                                                     ***********                                                                                                      
#                                                                                     .***,.                                                                                                           
#                                                                                       ,/////////      ,*                                                                                             
#                                                                                 .,    //////////                                                                                                     
#                                                                              *.       //////////                                                                                                     
#                                                                           *           ,////////*  .,***********,.                                                                                    
#                                                                        ,,              ///. .*****,.             .,**,                                                                               
#                                                                       ,                 ,***,                          ,*                                                                            
#                                                                                     ,**,                                  .*                                                                         
#                                                                                 .***                                         ,                                                                       
#                                                                         /%/ .***                                              *                                                                      
#                                                                       /@@@@@*                              .*/*.               .                                                                     
#                                                                      ,@@@@@@@*        *(%&&@@@@&%#(*.  #@@@@@@@@@@                                                                                   
#                                                                   ,*,&@@*(@@@@.%@@@@@@@@@@@@@@@@@@.,&@@@@@@@@@@@@@#                                                                                  
#                                                        ,**********.  &@@/./@@@@@@@@@@@@@@@@@@@@@.%@@@@@@@@&&@@@@@@#                                                                                  
#                                                                      (@.@@.&@@@@@@@@@@@@@@@@@@#@@@@@@, *&@@*@@@@@@//                                                                                 
#                                                                      @(&@@%*@@@@@@@@@@@@@@@@@@@@@@@,.@@@@@@%#@@@@@*@%                                                                                
#                                                                     %@,@@@%,@@@@@@@@@@@@@@@@@@@@@/.@@@@@@@@%,@@@@@@@@#                                                                               
#                                                                    /@,&@@@&.@@@@@@@@@@@@@@@@@@@@,@@@@@@@@@@%,@@@@@@@@@#                                                                              
#                                                                    @&*@@@@&.@@@@@@@@(@@@@@@@@@&.@@@@@@@@@@@%,@@@@@@@@@@                                                                              
#                                                                   /@/%@@@@@ @@@@@@@,@@@@@@@@@&.@@@@@@@@@@@@//@@@@@@@@@@/                                                                             
#                                                                   &@,@@@@@@ @@@@@@#@@@@@@@@@&.@@@@@@@@@@@@@.@@@@@@@@@@@@                                                                             
#                                                                   @%(@@@@@@ @@@@@@@@@@@@@@@@.@@@@@@@@@@@@@@,@@@@@@@@@@@@/                                                                            
#                                                                  @@%#@@@@@/ @@@@@@ &@@@@@@@ @@@/   .@@@@@@*&@@@@@@@@@@@@@,                                                                           
#                                                               (@@@@%#@@@@@  #@@@@ &@@@@@@@,&@@*     (@@@@@.@@@@@@@@@@@@@@@@@*      *                                                                 
#                                                           .(@@@@&@@@*@@@@&   @@@&.@@@@@@@.*@@%      ,@@@@%&@@@@@@@@@@@@@@@@@@@@@%,                                                                   
#                                                        %@@@@@@@@@@@@,@@@@@    ,@@,%@@@@# &@@@/      ,@@@@.@@@@@@@@@@@%&@@@@@@@@@@@(                                                                  
#                                                     #@//@@@@@@@@@@@@%#@@@@/      /(**, &@@@@@*      &@@@,&@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                                
#                                                   .* .@@@@@@@% .*/, *#@@@@@%, *(((((((((.&@@@&     #@@@/&@@@@@#      /@@@@@@@@@@@@@@*@,                                                              
#                                                      &@@@@@@& .,/* *##*(@@&*(((((((((((((,(@@@@,,#@@@@/&@@* ,(((((((((  @@@@@@@@@@@@@,.                                                              
#                                        *************, @%@@&*@@&&&&/*(.(((((((((((/,,*(#((((*@@@@@@@@@/%  /((((((*,,,((* &@@@@@@@@@@@@#                                                               
#                                                       (%@.@@@@@&&&&&(*(/(((((#* (&&&&%./*.(((/,**,. ,(((((((/ *(####(. /@@@@@@@@@@@.#%              #                                                
#                                                          @@@@@&&&&&&&*/,((((/*&&&&&&&&&/*/((((((((((((((,,#(, *(##%@@@@@@@@@@@@@@/(*@@@@@@@@@@@@@@@                                                
#                                                          @@@&&&&&&&&&%.,..*((&&&&&&&&&&&&&.(/(((((((((((/.#*.@@@@@@@@@@@@@@@@@@@@@@,  *&@@@@@@@@@@@@@                                                
#                                                             ,(%&&&&&&/*.  **&&&&&&&&&&&&&&(%,(((((((((( ((     @@@@@@@@@@@@@@@@@@@&                                                                  
#                                                   *(#((((((#(/*.  .//((((((,&&&&&&&&&&&&&&,((((((((((/,#,       *@@@&@@@@@@@@@@@%&                                                                   
#                                                   ((((((((((((((((((((/. ,(#,&&&&&&&&&&&&, (((((((((.(*          @&..@@@@&@@/@# ,                                                                    
#                                                    *#/..,.     ,/(#(((((((((/,,#&&&&&&%. *((((((((,*(        .../   /,*,*/  .                                                                        
#                                                        */(((#(((#(/.  .(#(((((((((((((((((((((((**#,* .              /                                                                               
#                                                                   *(#(((/, .(#((((((((((((((((.,#*,@% ....          &@#                                                                              
#                                                                         .*(##(/, .*/(((((*,.*((.%@@& .......        &@#                                                                              
#             .(#(,  ,@@@@&.                                                       .,/(((//. .#@@@@@% ..........                                                                                       
#            @@@@@@.,@@@@@@@@(                                                      .... @@@@@@@@@&  ..............                                                                                    
#           &@@@@@# @@@@@@@@@@@&                                                   ..... @@@@@@@, ....................                                                                                 
#        @@.@@@@@@ ,@@@@@@@ *@@@@*                                                ...... #@@@/  ......................                                                                                 
#      /@@@ @@@@@& %@@@@@@& ,. /@@@                                              ........ %@/..........................                                                                                
#     (@@.. @@@@@*.@@@@@@@ ,,,,, #@@%                                            ..........,..................      .                                                                                  
#    &@& ,,.&@@@@.(@@@@@@. .,,,,, *@@%                                       ,,  ..........................  .,,,,,,,,,,,.   ..                                                                        
#   &@@ ,,,./@@@@.%@@@@@..@@,.,,,,,.@@@.                                ,,,,,,,  ......... ..............  ,,,,,,,,,,,,,,,,,,, ,,,. .,,,,,,,..                                                         
#  %@@%.,,,, @@@@%.@@@& (@@@@/ ,,,,,*@@@/ ......                   .,,,,,,..,,,  ........ . ............ ,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,,,,,,,,,,,.                                                    
# .@@@.,,,,, .@@@@###%@@@@@@@@/ ,,,, @@@@/...............  ,,,,,,,,,,,,,,,,,,,,. ....... .,, .......... ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                                                  
#      ,,,,,      *@@@@@@@@@@@@(.... &@@@@/. .............. .,,,,,,,,,,,,,,,,,,,  ..... .,,,, .........,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                                                
#      ,,,,,        @@@@@@@@&..%@@@@@@@@@@@%& .............. .,,,,,,,,,,,,,,,,,,,  ...  ,,,,,. ....... ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.                                       .      
#      ,,,,,         %@@@@@,,@@@@@@@@@@@@@@@@, .............. ,,,,,,,,,,,,,,,,,,,,,  .,,,.*# ,,  .... ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                                      /,      
#      ,,,,,          *@@@@ @@@@@@@@@@@@@@@@@& ............... ,,,,,,,,,,,,,,,,,,,,, ,,, /###.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,                                     @@       
#      .,,,,            #@@ (@@@@@@@@@@@@@@@@@ ............... ,,,,,,,,,,,,,,,,,,,,, ,,,.*###.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  .,,,.                            ,@@@       
#       ,,,,              /@/ .,.#@*@@@@@@@@@@ ............... .,,,,,,,,,,,,,,,,,,,,..,,,.,#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,,,,,,,,,.                      %@@@@%       
#       .,,,,               *@@@@@@@@@@@@@@@@@ ............... .,,,,,,,,,,,,,,,,,,,,. ,,,,,,,,,,,,,,,,,,,,,,,,,,,,....,...   .,,,,,,,,,,,,,,,,,,,,,,,,,.  ,,,,,,,,,,,  &@#. //         (@@@@@@@/       
#        ,,,,                   (&@@@&(/..     ............... .,,,,,,,,              ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,   .,,,,,,.  .,,,,,,,,,,,,,,,,,,  .,,,,,,,,,,,,,, (@@@@, &@* &@@@@@@@@@@@%       
#        .,,,,                      ..        ................ ,,,                    ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,     ,,,,,,,   ,,,,,,,,,,,,,,,,,,,./@@@@@& *@# (@@@@@@@@@/       
#         ,,,,                      ...      ................                         .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,, ,,,...,,,,,,,,,,,,,,,,,,,,,,,,,.*@@@@@@@ &@( @@@@@@@@*       
#          ,,,,                      .......................                           ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.*@@@@@@@#&@@,@@@@@@@*       
#          .,,,                        ..................                              .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..,..,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, #@@@@@@@@@@@@@@@@@@/       
#           ,,,,                         ..........                                     ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .,  .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@@@@@@@@.       
#           ,,,,,                                                                       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,, ,,, .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, @@@@@@@@@@@@@@@@@        
#            ,,,,                                                                       .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@@@@@@&  ,#    
#             ,,,.                                                                       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,,..,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.*@@@@@@@@@@@@@@@/%@@     
#             ,,,,                                                                       .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .,. ,,,,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, @@@@@@@@@@@@@@@@@@      
#              ,,,,                                                                       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .,, ,,,,,,,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,, #@@@@@@@@@@@@@@@*       
#              .,,,.                                                                       ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,..@@@@@@@@@@@@@%  **     
#               ,,,,                                                                        ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, .,, ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, @@@@@@@@@@@@@@@%.      
#                ,,,,                                                                        ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, ,,. ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@&@&       
#                ,,,,.                                                                        ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  ,, .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@@%          
#                .,,,,                                                                         ,,,,,,,,,,,,,,,,,,,,,,,,,,,,.  ,,  ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@%&          
#                 ,,,,                                                                           ,,,,,,,,,,,,,,,,,,,,,,,,  .,.  ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, &@@@@@@@@@@(           
#                  ,,,,                                                                            ,,,,,,,,,,,,,,,,,,,  .,,  .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,. @@@@@@@@@@%            
#                  ,,,,,                                                                                 ..,,,..    .,,.  .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, (@@@@@@@@@#             
#                   ,,,,                                                                                    ...,.... .,#@#.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,. @@@@@@@@@#              
#                   .,,,,                                                                                     ,,,,,,,,.@@@@/ ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, (@@@@&&@@(               
#                    ,,,,,                                                                                    ,,,,,,,  .@@@@@@/ ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,     .@@@(                 
#                     ,,,,                                                                                    ,,,,,,      /@@@@@* ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,     ,@@*                   
#                     ,,,,,                                                                                    ,,,,.          (@@@& .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  .(( %#                      
#                      ,,,,                                                                                    .,,,,              #@@ .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,  /(#.                         
#                      .,,,,                                                                                     ,,,.          (((#(/,   ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.  (((*                          
#                       ,,,,,                                                                                      ,,,.      .(((((((((    ,,,,,,,,,,,,,,,,,,,,,,,,,.  *((((                           
#                        ,,,,                                                                                              , (((((((((       ,,,,,,,,,,,,,,,,,,,,,,  .((((((                           
#                        .,,,,                                                                                           .##*,/#(((((# ##,     ,,,,,,,,,,,,,,,,,,   ((((((((                           
#                         ,,,,,                                                                                           (#####(/*,,,.*###     .,,,,,,,,,,,,,     #((((((((                           
#                          ,,,,                                                                                            ,###############       .,,,,,,,,.      ,(((((((((,                          
#                          ,,,,,                                                                                         ,###(/.,*//(/(*.                          #((((((((( ,.                       
#                           ,,,,.                                                              (###########(*..       ,(#################*                         ,((((((((((,####/                   
#                            ,,,,                                                         ,(###(#(/,,/####################################*                     .##,((((((((((* /####*                 
#                            .,,,,                                                        (((((((((((((/  /#################################,                   (###(,,..   ,*########                 
#                             ,,,,.                                                        .#((((((((((((((. *#################################/                 ,##################* ,                
#                              ,,,,                                                         ,(((((((((((((((((  *###############################(                   ,/#########(/. (####               
#                              ,,,,,                                                         ,((((((((((((((((((* ,#############################,                    .#((/,,*/(######(*,/#*            
#                               ,,,,,                                                           /(##((((((((((((((.  (########################( ,(/                .,.##############(.####./(.         
#                               ,,,,,                                                                     /#((((((((#* .(##################* .((((.            ./#####(*../###########.,###.(####*     
#                                                                                                           (((((((((((#(/,    . (#######/(((#*  .,,**//((#################(, /#########(//#(,./###(   
#                                                                                                             .((((((((((((((((( (#######*    /#################################//#########(####(.###( 
#                                                                                                                  .*****,,.    /(/*,          *(((/,.,*(###################################/####//###.
#                                                                                                                                           /(((((((((((#(/,  ./(#############################*..,#### 
#                                                                                                                                          (((((((((((((((((((((((*   ,(############################(  
#                                                                                                                                            /##(((((((((((((((((((((#(*   ./######################,(  
#                                                                                                                                                 /((((((((((((((((((((((((#(.  *(#########((#####(    
#                                                                                                                                                   ,((((((((((((((((((((((((((((*.       *((.####     
#                                                                                                                                                     /(((((((((((((((((((((((((((((((((((((# (#*      
#                                                                                                                                                         /(#####((///((#(((((((((((((((((((,          
#                                                                                                                                                                          ./#((((((((((/              