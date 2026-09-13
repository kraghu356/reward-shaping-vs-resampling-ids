#!/usr/bin/env python3
"""
make_variants.py -- generate Elsevier, IEEE and Springer builds from paper.tex.

The body (Section 1 onward, plus appendices) is shared verbatim, so the three
variants cannot drift apart in content.  Only the preamble, the front matter
and a few float/bibliography conventions differ.
"""
import os
import re

MASTER = "paper.tex"
OUT = "variants"
os.makedirs(OUT, exist_ok=True)

src = open(MASTER, encoding="utf-8").read()

# ---- split the master ---------------------------------------------------
body = src[src.index(r"\section{Introduction}"):src.index(r"\bibliographystyle")]
appx = src[src.index(r"\appendix"):src.index(r"\end{document}")]
body = body.replace(appx, "")            # appendix handled separately

ABSTRACT = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
ABSTRACT = ABSTRACT.replace(r"\noindent", "").strip()
KEYWORDS = ("intrusion detection; class imbalance; reward shaping; "
            "class-balanced sampling; deep reinforcement learning; "
            "factorial ablation; NetFlow; UNSW-NB15")

COMMON_PKGS = r"""\usepackage{amsmath,amssymb,amsthm}
\usepackage{placeins}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{0.9}
\renewcommand{\textfraction}{0.07}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage[noend]{algpseudocode}
\usepackage{url}

\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{remark}{Remark}
\newtheorem{definition}{Definition}

\newcommand{\Real}{\mathbb{R}}
\newcommand{\Exp}{\mathbb{E}}
\newcommand{\RU}{\textsc{ru}}
\newcommand{\PU}{\textsc{pu}}
\newcommand{\PB}{\textsc{pb}}
\newcommand{\RB}{\textsc{rb}}
"""

BACKMATTER = r"""
\section*{CRediT authorship contribution statement}
\textbf{Raghavendra K:} Conceptualization, Methodology, Software, Formal
analysis, Investigation, Writing -- original draft, Writing -- review \&
editing. \textbf{Srividya Ramisetty:} Methodology, Validation, Writing --
review \& editing. \textbf{Manoj Kumar M:} Data curation, Validation, Writing
-- review \& editing. \textbf{M J Yogesh:} Software, Validation,
Visualization, Writing -- review \& editing.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

\section*{Data and code availability}
NF-UNSW-NB15-v2 is publicly available from the authors of
\cite{sarhan2022standard}. The preprocessing pipeline, the factorial runner,
the tuning harness with its validation protocol, the aggregated per-seed
results and the figure-generation scripts are available at
\url{https://github.com/kraghu356/CNN-LSTM-DQN-IDS-v2}.
"""

# strip the master's own backmatter from the shared body
for sec in ("CRediT authorship", "Declaration of competing",
            "\\section*{Funding}", "Data and code availability"):
    i = body.find("\\section*{" + sec) if not sec.startswith("\\") else body.find(sec)
    if i > 0:
        body = body[:i]
        break

TITLE = ("Reward Shaping or Resampling? Disentangling the Source of "
         "Rare-Class Gains in Deep Reinforcement Learning for "
         "Network Intrusion Detection")

AFFA = ("Department of Computer Science and Engineering (Artificial Intelligence), "
        "Maharaja Institute of Technology Mysore, Mandya, Karnataka, India")
AFFB = ("Department of Information Science and Engineering, "
        "CMR Institute of Technology, Bengaluru, Karnataka, India")
AFFC = ("Department of Computer Science and Engineering, "
        "Jyothy Institute of Technology, Bengaluru, Karnataka, India")
AFFD = "Vellore Institute of Technology, Vellore, Tamil Nadu, India"


# =====================================================================
# 1. ELSEVIER
# =====================================================================
els = r"""% ===== ELSEVIER (elsarticle) build =====
% Generated from paper.tex by make_variants.py -- do not edit directly.
% Submission: replace [review] with [final,5p,times,twocolumn] for production.
\documentclass[review,12pt]{elsarticle}
\usepackage{lineno}
""" + COMMON_PKGS + r"""
\journal{Computers \& Security}

\begin{document}

\begin{frontmatter}

\title{""" + TITLE + r"""}

\author[a]{Raghavendra K\corref{cor1}}
\ead{Raghavendrak_ai@mitmysore.in}
\cortext[cor1]{Corresponding author. ORCID 0000-0003-1256-1975.}
\author[b]{Srividya Ramisetty}
\author[c]{Manoj Kumar M}
\author[d]{M J Yogesh}

\affiliation[a]{organization={""" + AFFA + r"""}}
\affiliation[b]{organization={""" + AFFB + r"""}}
\affiliation[c]{organization={""" + AFFC + r"""}}
\affiliation[d]{organization={""" + AFFD + r"""}}

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\begin{keyword}
""" + KEYWORDS.replace("; ", " \\sep ") + r"""
\end{keyword}

\end{frontmatter}

\linenumbers

""" + body + r"""
\appendix
""" + appx.replace(r"\appendix", "") + BACKMATTER + r"""
\bibliographystyle{elsarticle-num}
\bibliography{refs}

\end{document}
"""

# =====================================================================
# 2. IEEE  (two-column: wide floats must become starred)
# =====================================================================
ieee_body = body
ieee_appx = appx.replace(r"\appendix", "")
# IEEEtran uses its own \cite and does not load natbib, so natbib-specific
# citation commands must be converted.  \citet{x} appears in running text as
# "Author [n]" in natbib; IEEEtran has no equivalent, so the author name is
# already written out in the body and a plain \cite is correct here.
for a, b in ((r"\citep{", r"\cite{"), (r"\citet{", r"\cite{")):
    ieee_body = ieee_body.replace(a, b)
    ieee_appx = ieee_appx.replace(a, b)
# widen the multi-column tables and figures for a two-column layout
for src_env, dst_env in ((r"\begin{table}[t]", r"\begin{table*}[t]"),
                         (r"\begin{table}[h]", r"\begin{table*}[t]"),
                         (r"\begin{table}[!ht]", r"\begin{table*}[t]"),
                         (r"\begin{table}[!htbp]", r"\begin{table*}[t]"),
                         (r"\end{table}", r"\end{table*}"),
                         (r"\begin{figure}[t]", r"\begin{figure*}[t]"),
                         (r"\end{figure}", r"\end{figure*}"),
                         (r"\begin{algorithm}[t]", r"\begin{algorithm}[t]")):
    ieee_body = ieee_body.replace(src_env, dst_env)
    ieee_appx = ieee_appx.replace(src_env, dst_env)

ieee = r"""% ===== IEEE (IEEEtran) build =====
% Generated from paper.tex by make_variants.py -- do not edit directly.
% Wide tables/figures are starred floats for the two-column layout.
\documentclass[journal,11pt]{IEEEtran}
""" + COMMON_PKGS + r"""
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{""" + TITLE + r"""}

\author{
Raghavendra~K,~\IEEEmembership{}
Srividya~Ramisetty,
Manoj~Kumar~M,
and~M~J~Yogesh
\thanks{R. Raghavendra is with the """ + AFFA + r""" (e-mail: Raghavendrak\_ai@mitmysore.in; ORCID 0000-0003-1256-1975).}
\thanks{S. Ramisetty is with the """ + AFFB + r""".}
\thanks{M. Kumar M is with the """ + AFFC + r""".}
\thanks{M. J. Yogesh is with """ + AFFD + r""".}
}

\maketitle

\begin{abstract}
""" + ABSTRACT + r"""
\end{abstract}

\begin{IEEEkeywords}
""" + KEYWORDS.replace("; ", ", ") + r"""
\end{IEEEkeywords}

\IEEEpeerreviewmaketitle

""" + ieee_body + r"""
\appendices
""" + ieee_appx + BACKMATTER + r"""
\bibliographystyle{IEEEtran}
\bibliography{refs}

\end{document}
"""

# =====================================================================
# 3. SPRINGER
#    sn-jnl.cls is distributed by Springer and is not in TeX Live, so this
#    build uses the article class configured to Springer's manuscript
#    conventions.  A one-line swap is documented at the top of the file.
# =====================================================================
spr = r"""% ===== SPRINGER build =====
% Generated from paper.tex by make_variants.py -- do not edit directly.
%
% Springer Nature distributes sn-jnl.cls with its author kit; it is not part
% of TeX Live, so this file uses the standard article class configured to
% Springer's manuscript conventions (double spacing, numbered citations,
% abstract + keywords front matter).  Springer accepts this for initial
% submission.  To switch to the official class once you have downloaded the
% author kit, replace the \documentclass line below with
%
%     \documentclass[sn-mathphys-num,iicol]{sn-jnl}
%
% and wrap the front matter in \abstract{...} / \keywords{...} as the kit
% describes; the body below needs no changes.
\documentclass[12pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
""" + COMMON_PKGS + r"""
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{setspace}
\doublespacing

\title{\textbf{""" + TITLE + r"""}}

\author{
Raghavendra K$^{1*}$ \and Srividya Ramisetty$^{2}$ \and
Manoj Kumar M$^{3}$ \and M J Yogesh$^{4}$
}
\date{}

\begin{document}
\maketitle

\noindent
$^{1}$""" + AFFA + r"""\\
$^{2}$""" + AFFB + r"""\\
$^{3}$""" + AFFC + r"""\\
$^{4}$""" + AFFD + r"""\\[4pt]
$^{*}$Corresponding author. E-mail: \texttt{Raghavendrak\_ai@mitmysore.in};
ORCID 0000-0003-1256-1975.

\vspace{10pt}
\noindent\textbf{Abstract}\quad """ + ABSTRACT + r"""

\vspace{8pt}
\noindent\textbf{Keywords}\quad """ + KEYWORDS + r"""

\vspace{10pt}

""" + body + r"""
\appendix
""" + appx.replace(r"\appendix", "") + BACKMATTER + r"""
\bibliographystyle{unsrtnat}
\bibliography{refs}

\end{document}
"""

for name, text in (("paper_elsevier.tex", els),
                   ("paper_ieee.tex", ieee),
                   ("paper_springer.tex", spr)):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT}/{name}  ({len(text)} chars)")
print("\nshared body:", len(body), "chars; appendix:", len(appx), "chars")
