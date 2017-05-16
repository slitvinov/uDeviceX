A wrapper for cgal [^1]

Installing cgal
===============

debian
------

``` {.bash}
sudo apt install libcgal-dev libcgal-qt5-dev -y
```

osx
---

``` {.bash}
brew install cgal
echo 'export CGAL_DIR=/usr/local/Cellar/cgal/4.9' >>~/.bash_profile
```

daint
-----

``` {.bash}
git clone https://github.com/Linuxbrew/brew.git ~/.linuxbrew
brew install cgal
```

Build geom-wrapper
==================

ubuntu
------

``` {.bash}
cmake . -DCMAKE_VERBOSE_MAKEFILE=ON
make
```

falcon
------

    PATH=$HOME/.linuxbrew/bin:$PATH CGAL_DIR=$HOME/.linuxbrew/Cellar/cgal/4.9 cmake . -DCMAKE_VERBOSE_MAKEFILE=ON
    make

daint
-----

    PATH=$HOME/.linuxbrew/bin:$PATH CGAL_DIR=$HOME/.linuxbrew/Cellar/cgal/4.9  cmake . -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

Linking with geom-wrapper
=========================

``` {.example}
GWRP_CXXFLAGS = -I../geom-wrapper
GWRP_LDFLAGS = -L${HOME}/prefix/cgal/lib64 -Wl,-rpath,${HOME}/prefix/cgal/lib64:${HOME}/prefix/pkgsrc/lib
GWRP_LIBS = ../geom-wrapper/libgeom-wrapper.a -lgmp -lCGAL
```

[^1]: <http://www.cgal.org>
