In addition to the already defined features in this chat, these are additional guidelines for this pipeline class:

0 - Let's name it "Ontology_Foundation_Builder" class:

1 - It should take two inputs:
    a - A JSON Hierarchy Structure: The hierarchy nested list of clusters with the original chunk IDs at the leaf level and the top-10 retrieved chunk ids at every level. These chunks are already embedded in the vector DB (see uploaded json hierarchy)
    b - A folder where one json for each cluster is stored, containing their baseline info: label, summary, keywords, candidate entities and candidate processes (see one uploaded cluster baseline json file for cluster 0.0.0.0.0.3.1)

2 - All intermediate generated json files should be placed in this structure:
    - <Ontolofy_Foundation_Dir> (given as an input parameter)
        - Entitities
            - Step_{n}
              {entity_ID}_step{n}.json
        - Relationships
            - Step_{n}
              {relationship_ID}_step{n}.json
        - Processes
            - Step_{n}
              {process_ID}_step{n}.json
        - Attributes
            - Step_{n}
              {attribute_ID}_step{n}.json

3 - A new progress bar line should start for every new <element>/<step> (i.e. Entities/Step_2)

4 - These files will be output and input to the next step as required

5 - The final data which should be stored and embed in the Vector DB should be also stored in these folders as the final step data. Only from there the final-final step would kick-off, picking that final data from the folders and saving / embedding in the VDB in the four corresponding collections (Entitties, Relationships, Processes, Attributes)

6 - Other pre-discussed features:
    - Multi parallel request to LLM from the main thread
    - Resilience to continue from the last save point in case of crash or any other disrruption
    - Ventor DB and LLM passed as object parameters ( see both already uploaded backend code)
    - Use the simple progress bar uploaded as a reference

7 - Any other feature we might have discussed and not it htis list