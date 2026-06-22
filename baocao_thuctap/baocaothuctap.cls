%% ============================================================
%%  BAOCAOTHUCTAP.CLS — Class file for Báo Cáo Thực Tập
%%  Trung tâm An ninh mạng
%%  Based on: extreport (a4paper, 14pt, oneside) via extsizes
%% ============================================================
\ProvidesClass{baocaothuctap}[2026/06/21 v1.0 Bao cao thuc tap class]
\LoadClass[a4paper,14pt,oneside]{extreport}

%% ---- Font & ngôn ngữ (XeLaTeX) ----
\RequirePackage{fontspec}
\setmainfont{Times New Roman}

%% ---- Geometry ----
\RequirePackage[top=2.5cm, bottom=2.5cm, left=3cm, right=3cm]{geometry}

%% ---- Packages cơ bản ----
\RequirePackage{amsmath,amssymb}
\RequirePackage{graphicx}
\RequirePackage[table]{xcolor}
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{array}
\RequirePackage{hyperref}
\RequirePackage{float}
\RequirePackage{pdfpages}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{caption}
\RequirePackage{multirow}

%% ---- Thuật toán ----
\RequirePackage[algoruled,linesnumbered]{algorithm2e}
\DontPrintSemicolon

%% ---- Đánh số hình theo chapter ----
\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}
\numberwithin{equation}{chapter}

%% ---- Tiếng Việt cho các caption ----
\renewcommand{\contentsname}{Mục lục}
\renewcommand{\listfigurename}{Danh sách hình vẽ}
\renewcommand{\listtablename}{Danh sách bảng biểu}
\renewcommand{\bibname}{Tài liệu tham khảo}
\renewcommand{\appendixname}{Phụ lục}
\renewcommand{\chaptername}{Chương}
\renewcommand{\figurename}{Hình}
\renewcommand{\tablename}{Bảng}
\renewcommand{\thechapter}{\arabic{chapter}}

%% ---- Header / Footer ----
\setlength{\headheight}{15.5pt}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}

%% ---- Section format ----
\titleformat{\chapter}[display]
  {\bfseries\large\centering}
  {\MakeUppercase{\chaptername\ \thechapter}}
  {1ex}
  {\titlerule\vspace{1ex}\bfseries\LARGE}

\titleformat{\section}{\bfseries\large}{\thesection}{1ex}{}
\titleformat{\subsection}{\bfseries\normalsize}{\thesubsection}{1ex}{}
\titleformat{\subsubsection}{\bfseries\normalsize}{\thesubsubsection}{1ex}{}

%% ---- Giãn dòng 1.5 ----
\onehalfspacing

%% ---- Hyperlink Setup ----
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}
\endinput
