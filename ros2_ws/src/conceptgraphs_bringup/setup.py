from glob import glob
from setuptools import find_packages, setup


package_name = "conceptgraphs_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config",
         glob("config/*.yaml") + glob("config/*.rviz")),
        (f"share/{package_name}/config/datasets", glob("config/datasets/*.yaml")),
        (f"share/{package_name}/config/layouts", glob("config/layouts/*.yaml")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/scripts", glob("scripts/*.py")),
        (f"share/{package_name}/sql", glob("sql/*.sql")),
        (f"share/{package_name}/web", glob("web/*")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="orc",
    maintainer_email="orc@example.com",
    description="ConceptGraphs visualization bringup",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "z_passthrough_filter = conceptgraphs_bringup.z_passthrough_filter:main",
            "sam3_crate_mapper = conceptgraphs_bringup.sam3_crate_mapper:main",
            "depth_to_color_align = conceptgraphs_bringup.depth_to_color_align:main",
            "convert_kilted_bag = conceptgraphs_bringup.convert_kilted_bag:main",
            "rear_person_filter = conceptgraphs_bringup.rear_person_filter:main",
            "registered_cloud_dynamic_filter = conceptgraphs_bringup.registered_cloud_dynamic_filter:main",
            "rgb_frame_extractor = conceptgraphs_bringup.rgb_frame_extractor:main",
            "crate_marker_publisher = conceptgraphs_bringup.crate_marker_publisher:main",
            "greenhouse_layout_markers = conceptgraphs_bringup.greenhouse_layout_markers:main",
            "extract_keyframes = conceptgraphs_bringup.keyframe_extractor:main",
            "keyframe_marker_publisher = conceptgraphs_bringup.keyframe_marker_publisher:main",
            "run_sam3_offline = conceptgraphs_bringup.sam3_offline:main",
            "project_observations_3d = conceptgraphs_bringup.observation_projection:main",
            "observation_3d_marker_publisher = conceptgraphs_bringup.observation_3d_marker_publisher:main",
            "associate_objects = conceptgraphs_bringup.object_association:main",
            "object_marker_publisher = conceptgraphs_bringup.object_marker_publisher:main",
            "enrich_semantics = conceptgraphs_bringup.semantic_enrichment:main",
            "build_hmsg_database = conceptgraphs_bringup.hmsg_database:main",
            "query_hmsg_database = conceptgraphs_bringup.hmsg_query:main",
            "query_hmsg = conceptgraphs_bringup.hmsg_retrieval:main",
            "query_hmsg_web = conceptgraphs_bringup.query_web:main",
        ],
    },
)
