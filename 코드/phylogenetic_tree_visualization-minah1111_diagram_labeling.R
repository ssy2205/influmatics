suppressPackageStartupMessages({
  library(ggtree); library(treeio); library(ape)
  library(dplyr);   library(stringr); library(ggplot2); library(scales)
})

## 1) 라벨 정규화
norm_label <- function(x){
  x %>%
    str_remove_all("[\"'\\(\\)\\[\\]]") %>%
    str_replace_all("[\\s/|:.-]+", "_") %>%
    str_replace_all("(__[A-Za-z0-9]+_)+$", "") %>%
    str_replace_all("_+", "_") %>%
    str_replace("^_", "") %>% str_replace("_$", "")
}

## 2) 파일 로드 (원본 코드로 복구됨)
tree <- read.nexus("H1N1_HA.nexus"); if (inherits(tree, "multiPhylo")) tree <- tree[[1]]
tree$tip.label <- norm_label(tree$tip.label)

meta_raw <- read.delim("H1N1_HA_.tsv", sep="\t", stringsAsFactors = FALSE)
stopifnot(all(c("seqName","legacy.clade") %in% names(meta_raw)))
meta <- meta_raw %>% transmute(label = norm_label(seqName),
                               clade = as.character(legacy.clade))

## 3) outlier 3개 제거
suspects_norm <- norm_label(c("A/Tianjin-baodi/1606/2018", "A/Yunnan-Mengzi/1462/2020", "A/Neimenggu-Hongshan/SWL2204/2023", "A/Hubei-Wujiagang/1324/2020", "A/Gansu-Xifeng/1194/2021", "A/Hebei-Yuhua/SWL1250/2012__H1N1v_", "A/Jiangsu/1/2011", "A/Shandong-Lanshan/SWL1898/2014", "A/Switzerland/5165/2010","A/Shandong/00204/2021", "A/Tianjin-baodi/1606/2018","A/Almaty/32/1998"))
to_drop     <- intersect(suspects_norm, tree$tip.label)
tree_clean  <- if (length(to_drop)) ape::drop.tip(tree, to_drop) else tree
cat(sprintf("# tips before/after: %d -> %d (removed %d)\n",
            length(tree$tip.label), length(tree_clean$tip.label),
            length(tree$tip.label) - length(tree_clean$tip.label)))

## 4) 메타 머지 & 팔레트
meta2 <- meta %>%
  mutate(clade_key = ifelse(is.na(clade) | clade=="", "Pre-clade strains", clade)) %>%
  filter(label %in% tree_clean$tip.label)

lvl <- c("Pre-clade strains", sort(setdiff(unique(meta2$clade_key), "Pre-clade strains")))
meta2$clade_key <- factor(meta2$clade_key, levels = lvl)
cols_other <- if (length(lvl) > 1) hue_pal()(length(lvl)-1) else character(0)
cols_named <- c("Pre-clade strains"="gray70", setNames(cols_other, lvl[-1]))

## 5) 기본 플롯
p <- ggtree(tree_clean, mrsd = "2025-06-01") %<+% meta2 +
  geom_tree(aes(color = clade_key), linewidth = 0.25, na.rm = TRUE) +
  scale_color_manual(values = cols_named, breaks = lvl, name = "Nextclade Clade") +
  theme_tree2() +  
  labs(title = sprintf("", length(tree_clean$tip.label)))

## =========================================================
## 6) [수정] 라벨 대신 도형으로 표시할 ID 입력 (입력 구간 분리)
## =========================================================

# (A) 동그라미(●)로 표시할 ID들: 여기에 목록을 넣으세요
ids_circle_raw <- c("69/2023","73/2023" ,"74/2023" ,"78/2023" ,"80/2023" ,"81/2023" ,"82/2023" ,"84/2023" ,"85/2023" ,"89/2023" ,"91/2023" ,"92/2023" ,"98/2023" ,"101/2023" ,"108/2023" ,"112/2023" ,"113/2023" ,"120/2023" ,"126/2023" ,"139/2023" ,"149/2023" ,"150/2023" ,"153/2023" ,"160/2023" ,"162/2023" ,"175/2023" ,"178/2023" ,"183/2023" ,"190/2023" ,"191/2023" ,"194/2024" ,"198/2023" ,"208/2024" ,"209/2023" ,"216/2025" ,"218/2025" ,"221/2025" ,"222/2025" ,"227/2025" ,"231/2025" ,"232/2025" ,"241/2025" ,"246/2025" ,"247/2025" ,"259/2025" ,"260/2025" ,"261/2025" ,"262/2025" ,"264/2024" ,"270/2025" ,"274/2025" ,"275/2025" ,"279/2025" ,"280/2025" ,"281/2024" ,"282/2024" ,"283/2025" ,"284/2024" ,"290/2024","291/2025" ,"293/2024" ,"295/2024" ,"299/2024" ,"300/2025" ,"302/2025" ,"304/2025" ,"305/2025" ,"307/2025" ,"310/2025" ,"313/2025" ,"317/2025" ,"318/2025","320/2024" ,"324/2025" ,"325/2025","327/2025" ,"328/2024" ,"330/2025" ,"331/2024" ,"332/2024" ,"334/2025" ,"336/2025" ,"337/2025" ,"338/2025" ,"339/2025" ,"340/2025" ,"341/2025" ,"342/2025" ,"343/2025" ,"344/2025" ,"345/2024" ,"346/2025" ,"347/2025" ,"348/2024" ,"350/2025" ,"351/2024","353/2025" ,"354/2025" ,"355/2025" ,"397/2023" ,"415/2025" ,"416/2023" ,"423/2023" ,"427/2023" ,"429/2023" ,"438/2025","439/2025" ,"440/2025" ,"442/2025","445/2025","448/2025" ,"449/2025" ,"68/2023","57/2023","67/2023") 

# (B) 네모(■)로 표시할 ID들: 여기에 목록을 넣으세요
ids_square_raw <- c("54/2023","352/2024", "287/2025", "181/2024", "132/2023" ,"450/2025" ,"83/2023" ,"65/2023") 

# -- 정규화 및 트리와 매칭 (자동 계산) --
ids_circle_norm <- unique(gsub("/", "_", ids_circle_raw))
ids_square_norm <- unique(gsub("/", "_", ids_square_raw))

tips        <- tree_clean$tip.label
hits_circle <- intersect(ids_circle_norm, tips)
hits_square <- intersect(ids_square_norm, tips)

## =========================================================
## 7) [수정] 백신주 설정 (세모 표시 + 라벨 On/Off)
## =========================================================
vacc_raw <- c("A/Victoria/4897/2022","A/Wisconsin/67/2022")
vacc_hits <- intersect(norm_label(vacc_raw), tips)

# 💡 백신 이름(라벨)을 표시할까요? (TRUE = 예 / FALSE = 아니오)
show_vaccine_label <- FALSE 

## =========================================================
## 8) [수정] 시각화 (도형 오버레이)
## =========================================================
Ntips <- length(tree_clean$tip.label)

p <- p +
  ## (1) 동그라미 그룹 (shape 16)
  geom_tippoint(
    data = function(d) subset(d, isTip & label %in% hits_circle),
    shape = 16, size = 2, color = "darkorange", na.rm = TRUE
  ) +
  
  ## (2) 네모 그룹 (shape 15)
  geom_tippoint(
    data = function(d) subset(d, isTip & label %in% hits_square),
    shape = 15, size = 2, color = "brown", na.rm = TRUE
  ) +
  
  ## (3) 백신주: 세모 (shape 17)
  geom_tippoint(
    data = function(d) subset(d, isTip & label %in% vacc_hits),
    shape = 17, size = 3.0, color = "red", na.rm = TRUE
  ) +
  
  ## x축 간격 및 레이아웃 (기존 유지)
  scale_x_continuous(breaks = scales::pretty_breaks(n = 20), expand = c(0, 0)) +
  coord_cartesian(ylim = c(-0.7, Ntips + 0.7), expand = 0, clip = "off") +
  theme(
    plot.margin = margin(t = 5, r = 26, b = 8, l = 5, unit = "mm"),
    legend.position = "none",
    legend.justification = c(0, 0.5),
    legend.direction = "vertical",
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 6),
    legend.key.width = unit(3, "mm"),
    legend.key.height = unit(3, "mm"),
    legend.box.margin = margin(t = 0, r = 2, b = 0, l = 6, unit = "mm")
  ) +
  guides(color = guide_legend(ncol = 1))

## (4) 백신 라벨 조건부 추가
if (show_vaccine_label) {
  p <- p + geom_tiplab(
    data = function(d) subset(d, isTip & label %in% vacc_hits),
    aes(label = label),
    size = 1.4, fontface = "bold", hjust = -0.2, 
    color = "black", align = TRUE, linesize = 0.18, na.rm = TRUE
  )
}

print(p)

## (선택) 저장
ggsave("NNA_branch_shapes_custom.pdf", p, width = 12, height = 30)