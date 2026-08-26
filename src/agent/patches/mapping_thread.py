"""Select depth, Hector, or hybrid occupancy as the authoritative map."""


def install():
    from rtnav.modules.mapping.mapping_thread import MappingThread
    from rtnav.modules.mapping.obstacle_map.obstacle_map import ObstacleMap
    from slam_occupancy_map import SlamOccupancyMap

    def new_obstacle_map(self):
        source = str(getattr(self.mapping_cfg, "map_source", "depth")).lower()
        if source == "depth":
            m = self.mapping_cfg
            return ObstacleMap(
                size=m.obstacle_map_size,
                pixels_per_meter=m.pixels_per_meter,
                config=m,
            )
        if source not in ("slam", "hybrid"):
            raise ValueError("mapping.map_source must be 'depth', 'slam', or 'hybrid'")
        m = self.mapping_cfg
        return SlamOccupancyMap(
            self.shared_state,
            size=m.obstacle_map_size,
            pixels_per_meter=m.pixels_per_meter,
            config=m,
            use_depth_obstacles=source == "hybrid",
        )

    MappingThread._new_obstacle_map = new_obstacle_map
    print("[patches] authoritative map source configurable: depth, SLAM, or hybrid")
