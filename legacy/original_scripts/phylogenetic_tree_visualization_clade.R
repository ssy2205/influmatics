## =========================
## HHA timetree + Nextclade → branch-only
## - outlier 3개 제거
## - 지정 ID + 백신주만 라벨
## - x축 딱 맞게 + 위/아래 여백
## - 범례: 트리 오른쪽(패널 밖), 살짝 띄움
## =========================
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

## 2) 파일 로드
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

## 5) 기본 플롯(브랜치만) + x축 표시
## 💡 'mrsd' (most recent sampling date)를 2025년경으로 설정
p <- ggtree(tree_clean, mrsd = "2025-06-01") %<+% meta2 +
  geom_tree(aes(color = clade_key), linewidth = 0.25, na.rm = TRUE) +
  scale_color_manual(values = cols_named, breaks = lvl, name = "Nextclade Clade") +
  theme_tree2() +   ## 💡 theme_tree2()에서 변경
  labs(title = sprintf("",
                       length(to_drop)))

## 6) 라벨 나올 대상(정확 일치만)
ids_raw <- c("69/2023","73/2023" ,"74/2023" ,"78/2023" ,"80/2023" ,"81/2023" ,"82/2023" ,"83/2023" ,"84/2023" ,"85/2023" ,"89/2023" ,"91/2023" ,"92/2023" ,"98/2023" ,"101/2023" ,"108/2023" ,"112/2023" ,"113/2023" ,"120/2023" ,"126/2023" ,"132/2023" ,"139/2023" ,"149/2023" ,"150/2023" ,"153/2023" ,"160/2023" ,"162/2023" ,"175/2023" ,"178/2023" ,"181/2024" ,"183/2023" ,"190/2023" ,"191/2023" ,"194/2024" ,"198/2023" ,"208/2024" ,"209/2023" ,"216/2025" ,"218/2025" ,"221/2025" ,"222/2025" ,"227/2025" ,"231/2025" ,"232/2025" ,"241/2025" ,"246/2025" ,"247/2025" ,"259/2025" ,"260/2025" ,"261/2025" ,"262/2025" ,"264/2024" ,"270/2025" ,"274/2025" ,"275/2025" ,"279/2025" ,"280/2025" ,"281/2024" ,"282/2024" ,"283/2025" ,"284/2024" ,"287/2025" ,"290/2024","291/2025" ,"293/2024" ,"295/2024" ,"299/2024" ,"300/2025" ,"302/2025" ,"304/2025" ,"305/2025" ,"307/2025" ,"310/2025" ,"313/2025" ,"317/2025" ,"318/2025","320/2024" ,"324/2025" ,"325/2025","327/2025" ,"328/2024" ,"330/2025" ,"331/2024" ,"332/2024" ,"334/2025" ,"336/2025" ,"337/2025" ,"338/2025" ,"339/2025" ,"340/2025" ,"341/2025" ,"342/2025" ,"343/2025" ,"344/2025" ,"345/2024" ,"346/2025" ,"347/2025" ,"348/2024" ,"350/2025" ,"351/2024" ,"352/2024" ,"353/2025" ,"354/2025" ,"355/2025" ,"397/2023" ,"415/2025" ,"416/2023" ,"423/2023" ,"427/2023" ,"429/2023" ,"438/2025","439/2025" ,"440/2025" ,"442/2025","445/2025","448/2025" ,"449/2025" ,"450/2025" ,"54/2023","65/2023","68/2023","57/2023","67/2023")
ids_norm <- unique(gsub("/", "_", ids_raw))
tips     <- tree_clean$tip.label
id_hits  <- intersect(ids_norm, tips)

## 7) 백신주
vacc_hits <- intersect(norm_label(c("A/Victoria/4897/2022","A/Wisconsin/67/2022")), tips)
## 8) 라벨/마커 오버레이 + 여백 + 범례 오른쪽(패널 밖)
Ntips <- length(tree_clean$tip.label)
p <- p +
  geom_tiplab(
    data = function(d) subset(d, isTip & label %in% id_hits),
    aes(label = label),
    size = 1.2, hjust = 0, align = TRUE, linesize = 0.18, na.rm = TRUE
  ) +
  geom_tippoint(
    data = function(d) subset(d, isTip & label %in% vacc_hits),
    shape = 4, size = 1.4, stroke = 0.85, color = "black", na.rm = TRUE
  ) +
  geom_tiplab(
    data = function(d) subset(d, isTip & label %in% vacc_hits),
    aes(label = label),
    size = 1.4, fontface = "bold", hjust = 0, align = TRUE, linesize = 0.18,
    color = "black", na.rm = TRUE
  ) +
  
  ## 💡 x축 간격 조절 (scale_x_continuous 사용)
  scale_x_continuous(
    breaks = scales::pretty_breaks(n = 20), ## 8개 정도의 "깔끔한" 눈금 자동 생성
    expand = c(0, 0) ## x축 여백 0 ("딱 맞게")
  ) +
  
  ## y축 위/아래 여백 + x축 딱 맞게 (clip 끔)
  coord_cartesian(ylim = c(-0.7, Ntips + 0.7), expand = 0, clip = "off") +
  theme(
    ## 오른쪽에 범례 공간 확보 (트리와 안 겹치게 r 마진만 키움)
    plot.margin = margin(t = 5, r = 26, b = 8, l = 5, unit = "mm"),
    ## 🔸 범례를 '오른쪽'으로 (패널 밖, 기본 위치)
    legend.position = "right", # <-- "none"을 "right"로 변경하여 범례 표시
    legend.justification = c(0, 0.5), # <-- 주석 해제하여 설정 적용
    legend.direction = "vertical", # <-- 주석 해제하여 설정 적용
    legend.title = element_text(size = 7), # <-- 주석 해제하여 설정 적용
    legend.text = element_text(size = 6), # <-- 주석 해제하여 설정 적용
    legend.key.width = unit(3, "mm"), # <-- 주석 해제하여 설정 적용
    legend.key.height = unit(3, "mm"), # <-- 주석 해제하여 설정 적용
    legend.box.margin = margin(t = 0, r = 2, b = 0, l = 6, unit = "mm") # <-- 주석 해제하여 설정 적용
  ) +
  guides(color = guide_legend(ncol = 1))

print(p)

## (선택) 저장
ggsave("NNA_branch_only_labels_exact_legendRightOutside.pdf", p, width = 1200, height = 3000)