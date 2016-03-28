FROM radioastro/python

MAINTAINER <sphemakh@gmail.com>

RUN apt-get update && apt-get install -y python-numpy \
    python-scipy \
    python-pyfits \
    python-matplotlib

# Not adding GPU depencies for now

ADD . /tmp/moresane

RUN cd /tmp/moresane && pip install .

CMD /usr/bin/runsane
