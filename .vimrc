## 1. Installer les plugins nécessaires avec vim-plug

```vim
call plug#begin('~/.vim/plugged')
Plug 'morhetz/gruvbox'
Plug 'itchyny/lightline.vim'
Plug 'preservim/nerdtree'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'mg979/vim-visual-multi'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
call plug#end()
```

## 2. Configurer des raccourcis similaires à VS Code

```vim
nnoremap <C-s> :w<CR>
Plug 'preservim/nerdcommenter'
nnoremap <C-\\> :vsplit<CR>
nnoremap <C-h> <C-w>h
nnoremap <C-l> <C-w>l
nnoremap <C-b> :NERDTreeToggle<CR>

" Supprimer une ligne : Ctrl+Shift+K
nnoremap <C-S-k> dd

" Dupliquer une ligne : Ctrl+D
nnoremap <C-d> yyp

" Sélectionner toutes les occurrences : Ctrl+Shift+L
nnoremap <C-S-l> :bufdo %s//g<Left><Left>

" Ajouter un caret (multi-cursors) : Ctrl+Alt+Click avec vim-visual-multi

" Déplacer une ligne : Alt+Up/Down
nnoremap <A-Up> :m-2<CR>==
nnoremap <A-Down> :m+1<CR>==

" Aller à la déclaration : Alt+Right
nnoremap <A-Right> :CocDefinition<CR>

" Déplacer mot par mot : Ctrl+Shift+Flèche
nnoremap <C-S-Right> w
nnoremap <C-S-Left> b
```

## 3. Configurer CoC pour l'auto-complétion
```vim
let g:coc_global_extensions = ['coc-tsserver', 'coc-pyright', 'coc-clangd']
```
