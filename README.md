**MUNDO: Protein Function Prediction Embedded in a Multi-species World (ISMB 2021 Submission)**


**Welcome to MUNDO, the Munk-based Unifier of Networks from Diverse Origanisms**

MUNDO has four main steps:
  * Network Processing
  * Landmark Selection
  * Co-embeddings
  * Function Prediction

In order for each network to be processed successfully, the name of each organism
and the name of the database where its interaction file was downloaded is needed.
If the name of either organism or the database is not recognized, network processing
will **fail**

A list of supported databases is given below:
  * BIOGRID
  * BIOPLEX
  * DIP
  * GENEMANIA
  * GIANT
  * HUMANNET
  * REACTOME
  * STRING
  * *LEGACY_BIOGRID* (internal Tufts BCB group format)

A mapping of supported organism UniProt IDs to their common names is given below:
  * ARATH : Arabidopsis thaliana (Mouse-Ear Cress)
  * CAEEL : Caenorhabditis elegans (Worm)
  * CANLF : Canis lupus familiaris (Dog)
  * CHICK : Gallus gallus domesticus (Chicken)
  * DICTY : Dictyostelium discoideum (Slime Mold)
  * DROME : Drosophila Melanogaster (Fly)
  * HUMAN : Homo Sapiens (Human)
  * MOUSE : Mus musculus (Mouse)
  * PIG   : Sus scrofa (Pig)
  * RAT   : Rattus norvegicus (Rat)
  * SCHPO : Schizosaccharomyces pombe (Fission Yeast)
  * YEAST : Saccharomyces cerevisiae (Baker's Yeast)

Once the networks have been processed, several BLASTP files will be saved to the directory
specified by the user. They will look like ```HUMAN_source1.txt```, ```MOUSE_target3.txt```, etc

After locating these files, please do the following:
  * Visit NCBI's BLASTP tool at ```https://blast.ncbi.nlm.nih.gov/Blast.cgi?```
  * On the search page, in the "Enter *Query* Sequence" box, select "Choose File". Select ```<organism name>_source_<n>.txt``` (n is some multiple of 500)
  * Select the "Align two or more sequences" checkbox
  * In the "Enter *Subject* Sequence" box, select "Choose File". Select ```<organism name>_target_<m>.txt``` (m is some multiple of batch size). Note that m may or may not equal n
  * Hit the "BLAST" button at the bottom of the page
  * After the search results load, download them as a **TEXT** file.
  * Repeat steps 4-6 with ```<organism name>_source_<m>.txt``` as the "*Subject* Sequence" and ```<organism name>_target_<n>.txt``` as the 
	"*Subject* Sequence" (reciprocal direction)
  * Repeat steps 4-7 with each source and target file pair as needed (reccomended is at least 2 pairs, meaning 6 total BLASTP queries). Note that files with lower numbers (```source1.txt```, ```source2.txt```) on average have **more** landmarks than higher number files
  * Run ```landmark_selection.py``` with the filepath to each pair of BLASTP results files. Note that ```landmark_selection.py``` **appends**the landmarks found in each pair to the existing ```reciprocal_best_hits.txt``` file



