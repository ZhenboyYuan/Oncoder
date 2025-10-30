# Preprocessing the PRAD dataset from TCGA
metadata <- read.table("/path/to/Metadata.tsv",sep = '\t',header = TRUE)
file_paths_tumor <- list.files(path = "/path/to/PRAD/Primary Tumor", pattern = "*.txt", full.names = TRUE)
tumor <- lapply(file_paths_tumor, function(x){read.table(x, header = FALSE, sep = "\t",row.names = 1)})
tumor <- do.call(cbind, tumor)
colnames(tumor) <-metadata$cases.0.submitter_id[match(basename(file_paths_tumor), metadata$file_name)]

file_paths_normal <- list.files(path = "./PRAD/Solid Tissue Normal", pattern = "*.txt", full.names = TRUE)
normal <- lapply(file_paths_normal, function(x){read.table(x, header = FALSE, sep = "\t",row.names = 1)})
normal <- do.call(cbind, normal)
colnames(normal) <- metadata$cases.0.submitter_id[match(basename(file_paths_normal), metadata$file_name)]

write.table(tumor,"./tumor_PRAD.tsv",sep = "\t",quote = FALSE)
write.table(normal, "./normal_PRAD.tsv",sep = "\t",quote = FALSE)
