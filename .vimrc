" .vimrc configuration file for Vim/Neovim

" Installer les plugins nécessaires avec vim-plug
call plug#begin('~/.vim/plugged')
Plug 'morhetz/gruvbox'
Plug 'itchyny/lightline.vim'
Plug 'preservim/nerdtree'
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'mg979/vim-visual-multi'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
call plug#end()

" Configurer des raccourcis similaires à VS Code
nnoremap <C-s> :w<CR>
nnoremap <C-\\> :vsplit<CR>
nnoremap <C-h> <C-w>h
nnoremap <C-l> <C-w>l
nnoremap <C-b> :NERDTreeToggle<CR>
nnoremap <C-S-k> dd
nnoremap <C-d> yyp
nnoremap <C-S-l> :bufdo %s//g<Left><Left>
nnoremap <A-Up> :m-2<CR>==
nnoremap <A-Down> :m+1<CR>==
nnoremap <A-Right> :CocDefinition<CR>
nnoremap <C-S-Right> w
nnoremap <C-S-Left> b

" Configurer CoC pour l'auto-complétion
let g:coc_global_extensions = ['coc-tsserver', 'coc-pyright', 'coc-clangd']
