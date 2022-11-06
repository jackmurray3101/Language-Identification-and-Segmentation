from pyannote.core import Timeline, Segment, Annotation
from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationCoverage, SegmentationPurity

'''
precision = SegmentationPrecision()
reference = Timeline()
hypothesis = Timeline()
reference.add(Segment("0", "1.5"))
reference.add(Segment(1.5, 2))
reference.add(Segment(2, 3))
reference.add(Segment(3, 4))
hypothesis.add(Segment(0, 1))
hypothesis.add(Segment(1, 2))
hypothesis.add(Segment(2, 3))
hypothesis.add(Segment(3, 4))
print(reference)
print(hypothesis)

p = precision(reference, hypothesis)
print(p)'''

reference = Annotation()
hypothesis = Annotation()

reference[Segment(0, 1)] = "en"
reference[Segment(1, 3)] = "fr"

hypothesis[Segment(0, 1)] = "en"
hypothesis[Segment(1, 2)] = "fr"
hypothesis[Segment(2, 3)] = "fr"

coverage = SegmentationCoverage()
cov = coverage(reference, hypothesis)

purity = SegmentationPurity()
pur = purity(reference, hypothesis)

print("Purity")
print(pur)

print("Coverage")
print(cov)
