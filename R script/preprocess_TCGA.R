# Preprocessing the LIHC dataset from TCGA
metadata <- read.table("~/software/DataMiner/LIHC/Metadata.tsv",sep = '\t',header = TRUE)

file_paths_tumor <- list.files(
    path = "~/software/DataMiner/LIHC/Primary Tumor", 
    pattern = "*.txt", 
    full.names = TRUE)
tumor <- lapply(file_paths_tumor, function(x){
    read.table(x, header = FALSE, sep = "\t",row.names = 1)})
tumor <- do.call(cbind, tumor)

colnames(tumor) <- metadata$cases.0.submitter_id[match(basename(file_paths_tumor), metadata$file_name)]

file_paths_normal <- list.files(
    path = "~/software/DataMiner/LIHC/Solid Tissue Normal", 
    pattern = "*.txt", 
    full.names = TRUE)
normal <- lapply(file_paths_normal, function(x){
    read.table(x, header = FALSE, sep = "\t",row.names = 1)})
normal <- do.call(cbind, normal)

colnames(normal) <- metadata$cases.0.submitter_id[match(basename(file_paths_normal), metadata$file_name)]
write.table(tumor,"~/software/Oncoder_2/STAR_PROTOCOL/data/tumor_LIHC.tsv",sep = "\t",quote = FALSE)
write.table(normal,"~/software/Oncoder_2/STAR_PROTOCOL/data/normal_LIHC.tsv",sep = "\t",quote = FALSE)



# Preprocessing the PRAD dataset from TCGA
metadata <- read.table("~/software/DataMiner/PRAD/Metadata.tsv",sep = '\t',header = TRUE)

file_paths_tumor <- list.files(
    path = "~/software/DataMiner/PRAD/Primary Tumor", 
    pattern = "*.txt", 
    full.names = TRUE)
tumor <- lapply(file_paths_tumor, function(x){
    read.table(x, header = FALSE, sep = "\t",row.names = 1)})
tumor <- do.call(cbind, tumor)

colnames(tumor) <- metadata$cases.0.submitter_id[match(basename(file_paths_tumor), metadata$file_name)]

file_paths_normal <- list.files(
    path = "~/software/DataMiner/PRAD/Solid Tissue Normal", 
    pattern = "*.txt", 
    full.names = TRUE)
normal <- lapply(file_paths_normal, function(x){
    read.table(x, header = FALSE, sep = "\t",row.names = 1)})
normal <- do.call(cbind, normal)

colnames(normal) <- metadata$cases.0.submitter_id[match(basename(file_paths_normal), metadata$file_name)]

write.table(tumor,"~/software/Oncoder_2/STAR_PROTOCOL/data/tumor_PRAD.tsv",sep = "\t",quote = FALSE)
write.table(normal,"~/software/Oncoder_2/STAR_PROTOCOL/data/normal_PRAD.tsv",sep = "\t",quote = FALSE
