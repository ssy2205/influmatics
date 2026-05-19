#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  make_option("--tree", type = "character", help = "Input tree file"),
  make_option("--metadata", type = "character", default = NULL, help = "Metadata TSV"),
  make_option("--out", type = "character", help = "Output image path"),
  make_option("--label-samples", type = "character", default = NULL, help = "Comma-separated sample labels"),
  make_option("--vaccine-strains", type = "character", default = NULL, help = "Comma-separated vaccine strain labels"),
  make_option("--show-legend", action = "store_true", default = FALSE),
  make_option("--show-clade-labels", action = "store_true", default = FALSE)
)

opts <- parse_args(OptionParser(option_list = option_list))

if (is.null(opts$tree) || is.null(opts$out)) {
  stop("--tree and --out are required")
}

message("Unified tree visualizer scaffold.")
message("Input tree: ", opts$tree)
message("Output: ", opts$out)
message("TODO: migrate ggtree plotting logic from legacy/original_scripts.")
